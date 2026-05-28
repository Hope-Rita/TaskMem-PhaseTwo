# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
import random
import torch

from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import openai
from time import sleep
from functools import wraps
import threading
import os
from json_repair import repair_json, loads

def timeout(seconds=300, default=None, raise_err=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                if raise_err:
                    raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")
                else:
                    return default
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

temp = 1e-6

# API config path can be set via env var M3_AGENT_API_CONFIG.
# Falls back to configs/api_config.json under cwd.
_API_CONFIG_PATH = os.environ.get("M3_AGENT_API_CONFIG", "configs/api_config.json")

client = {}
if os.path.exists(_API_CONFIG_PATH):
    config = json.load(open(_API_CONFIG_PATH))
    for model_name in config.keys():
        if isinstance(config[model_name], list):
            client[model_name] = [openai.AzureOpenAI(
                azure_endpoint=conf["azure_endpoint"],
                api_version=conf["api_version"],
                api_key=conf["api_key"],
            ) for conf in config[model_name]]
        else:
            client[model_name] = openai.AzureOpenAI(
                azure_endpoint=config[model_name]["azure_endpoint"],
                api_version=config[model_name]["api_version"],
                api_key=config[model_name]["api_key"],
            )
else:
    print(f"[WARN] m3_agent: API config not found at {_API_CONFIG_PATH}; "
          "LLM-based evaluation will fail. Set M3_AGENT_API_CONFIG env var to enable.")
FACE_PATTERN = re.compile(r"\bface[ _]\d+\b", re.IGNORECASE)

USER_PROMPT_EPISODIC_VIDEO = """You are provided with a video, a description of its preceding segment, and a generated candidate [Description] for the remaining portion.
Your task is to evaluate:
1. Whether the candidate description is factually accurate based only on visual content and subtitles (ignore audio).
2. Whether it connects coherently and naturally with the preceding description, without using transition words such as "continue".
For any spoken content, verify it solely against the displayed subtitles and disregard audio information.
Assign exactly one label:
1: Correct — The description that meets all of the above criteria.
0: Incorrect — Any description that fails to meet the above criteria.

Output Requirements: Return the result in the following valid JSON format only. Do not generate anything else.
{{
    "correctness_rationale": "Short explanation for marking this description as 1 or 0",
    "correctness": 1 or 0
}}

The description of the preceding segment:
{preceding_json}

The [Description] to verify:
{blocks_text}
""".strip()

USER_PROMPT_EPISODIC_TEXT = """
You are given the [Context] and a candidate description that are describing new events.

Your task is to evaluate whether the candidate description satisfies the following conditions.

Return label=0 if any condition is satisfied, else 1:
(1) The description repeats any atomic fact already present in the [Context].
(2) It includes any mention of bounding boxes, coordinates, or detection boxes (e.g., "bounding box", "bbox", "x1,y1,x2,y2", "rectangle box around").
(3) It contains meta phrases like: "subtitles said", "the subtitles say", "subtitle reads", "subtitle says", or "according to the subtitles".
(4) The quoted speech contains transcript-style speaker labels like "<face_id> says "<face_id>: Good"" inside quoted dialogue.
(5) It includes conclusion-based or context-setting statements such as "this video ends with..." or "based on previous videos".

Output Requirements: Return the result in the following valid JSON format only. Do not generate anything else.

{{
    "label_rationale": "Short explanation for marking this description as 1 or 0",
    "label": 1 or 0
}}

[Context]:
{preceding_json}

candidate description to verify:
{blocks_text}

""".strip()

HELPFULNESS_EPISODIC_PROMPT = """
You are given two [Description] and some example questions.
Based on the focus of the example questions, your task is to evaluate which description contains information that would be more useful for answering similar questions.
Output the ID of the more useful description. If both descriptions are equally useful (a tie), output -1.
- A set of example questions: {example_questions}
- Multiple groups of [Description]:
{blocks_text}
Return exactly one JSON object:
{{
    "more_useful_rationale": "Briefly introduce the reasons for making this judgment",
    "more_useful": "ID of the more useful description or -1"
}}
"""

MAX_RETRIES = 5
def get_response_with_retry(model, messages, timeout=30):
    for i in range(MAX_RETRIES):
        try:
            if isinstance(client[model], list):
                selected_model = random.choice(client[model])
            else:
                selected_model = client[model]
            if model in ["gemini-2.5-flash", "gemini-2.5-pro"]:
                extra_body={
                    "thinking": {
                        "include_thoughts": True,
                        "budget_tokens": 128
                    }
                }
                response = selected_model.chat.completions.create(model=model, messages=messages, temperature=temp, timeout=timeout, extra_body=extra_body, max_tokens=8192)
            else:
                response = selected_model.chat.completions.create(model=model, messages=messages, temperature=temp, timeout=timeout, max_tokens=8192)
            return response.choices[0].message.content
        except Exception as e:
            print("Failed to get response:", model, e)
            sleep(5)
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def generate_messages(inputs):
    messages = []
    messages.append(
        {"role": "system", "content": "You are an expert in video understanding."}
        )
    content = []
    for input in inputs:
        if input["type"] == "text":
            content.append(input)
        elif input["type"] == "video":
            base64_video = base64.b64encode(open(input["video"], "rb").read()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:video/mp4;base64,{base64_video}"},
                }
                )
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages

import numpy as np

def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # Optional: make bytes printable
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    return x  # already JSON-friendly

def _atomic_json_dump(payload, file_path):
    base, ext = os.path.splitext(file_path)
    final_path = file_path

    if os.path.exists(final_path):
        i = 1
        while True:
            candidate = f"{base}_{i}{ext}"
            if not os.path.exists(candidate):
                final_path = candidate
                break
            i += 1

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)

class MultiRewardEvaluator:
    def __init__(
        self,
        model_name_text="gpt-4o-2024-11-20",
        model_name_video="gemini-2.5-flash",
        tokenizer=None,
        max_response_length=8196,
        think_length_threshold=1600,
        memory_length_threshold=5,
        memory_token_threshold=25,
        question_path=None,
    ):
        self.model_name_text = model_name_text
        self.model_name_video = model_name_video
        self.think_start_ids = [151667,]
        self.think_end_ids = [151668,]
        self.memory_length_threshold = memory_length_threshold
        self.think_length_threshold = think_length_threshold
        self.max_response_length = max_response_length
        self.memory_token_threshold = memory_token_threshold
        self.tokenizer = tokenizer
        self.example_questions = []

        # question_path is only required when LLM-judge evaluation is invoked
        # (validation / test). Pure DPO training without validation can omit it.
        if question_path is not None:
            with open(question_path) as f:
                for line in f.readlines():
                    self.example_questions.append(line.strip())

        self.prompts = {
            "correctness": USER_PROMPT_EPISODIC_VIDEO,
            "label": USER_PROMPT_EPISODIC_TEXT,
        }

        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
    
    def find_pattern(self, seq, pat):
        if isinstance(pat, int) or (isinstance(pat, list) and len(pat) == 1):
            target = pat if isinstance(pat, int) else pat[0]
            seq_tensor = torch.tensor(seq)
            return (seq_tensor == target).nonzero(as_tuple=True)[0].tolist()
        
        # fallback to slow version for multi-token patterns
        n, m = len(seq), len(pat)
        return [i for i in range(n - m + 1) if seq[i:i + m] == list(pat)]
    def get_safe_values(self, source_dict, key, default_val=None):
        val = source_dict.get(key, {})
        if isinstance(val, dict):
            return list(val.values())
        print(f"[Error] Key '{key}' expected dict but got {type(val)}. Content: {str(val)[:100]}")
        return default_val if default_val is not None else []
    def right_trim(self, ids, pad_token_id):
        i = len(ids)
        while i > 0 and ids[i - 1] == pad_token_id:
            i -= 1
        return ids[:i]
    def _invalid_face_tag(self, desc: str) -> bool:
        if re.search(r'<face>', desc, re.IGNORECASE):
            return True

        suspicious_pattern = re.compile(r'face[\s_]*\d+', re.IGNORECASE)

        for m in suspicious_pattern.finditer(desc):
            s, e = m.span()
            token = m.group()  

            if token != f"face_{token.split('_')[-1]}" or not re.fullmatch(r'face_\d+', token):
                if not re.fullmatch(r'face_\d+', token):
                    return True

            has_brackets = (s > 0 and desc[s-1] == '<' and e < len(desc) and desc[e] == '>')
            if not has_brackets:
                return True

        return False

    def build_item(self, original_ids, extra_info):

        video_path = extra_info.get("video_path")
        preceding_description = extra_info.get("preceding_description", "")

        if hasattr(original_ids, "detach"):
            original_ids = original_ids.detach().cpu().tolist()

        valid_ids = self.right_trim(list(original_ids), self.pad_id)
        txt = self.tokenizer.decode(original_ids, skip_special_tokens=True)

        start_think_i = self.find_pattern(valid_ids, self.think_start_ids)
        end_think_i = self.find_pattern(valid_ids, self.think_end_ids)
        no_think = True if not end_think_i else False
        if no_think or len(end_think_i) > 1:
            decoded_info = (valid_ids, self.max_response_length, None, (0, len(valid_ids)), txt, [])
            print(f"[Warning] No thinking parsing found")
            return decoded_info, {}
        if start_think_i and end_think_i:
            think_len = max(0, end_think_i[0] - (start_think_i[0] + len(self.think_start_ids)))
        elif end_think_i:
            think_len = max(0, end_think_i[0])
        think_idx = (start_think_i[0] if start_think_i else 0, end_think_i[0] if end_think_i else 0)

        try:
            memory_key = "description"
            memory_desc = json.loads(txt.split("</think>")[-1].strip().strip("```json").strip())
            if not isinstance(memory_desc, dict) or memory_key not in memory_desc.keys() or len(memory_desc.keys()) > 1:
                decoded_info = (valid_ids, self.max_response_length, None, think_idx, txt, [])
                return decoded_info, {}
            val = memory_desc[memory_key]
            descs = [val if isinstance(val, str) else str(val)]
            start_idx = think_idx[1]
            spans = [(start_idx, len(valid_ids))]
        except Exception:
            print(f"Error parsing JSON")
            decoded_info = (valid_ids, self.max_response_length, None, think_idx, txt, [])
            return decoded_info, {}

        if len(spans) != len(descs) or len(descs) == 0:
            decoded_info = (valid_ids, think_len, None, think_idx, txt, spans)
            return decoded_info, {}

        group = {
                    "group_id": "0",
                    "descs": descs[:self.memory_length_threshold],
                    "video_path": video_path,
                    "preceding_description": preceding_description,
                }
        decoded_info = (valid_ids, think_len, descs, think_idx, txt, spans)
        return decoded_info, group

    def calculate_item_reward(self, decoded_info, task_results, info):
        ids, think_len, descs, think_idx, txt, spans = decoded_info
        info.update({
                "ids": ids,
                "think_len": think_len,
                "memory_token_length": max(0, len(ids) - think_len),
                "group_type": None,
                "num_correct": 0,
                "num_wrong": 0,
                "fail_parsing": 0 if descs is not None else 1,
                "invalid_face_list": [],
                "correctness_list": [],
                "ori_correctness_list": [],
                "redundancy_list": [],
                "valid_list": [],
                "memory_token_list": [],
                "memory_count": [],
                "has_face_list": [],
                "cot_correctness": [],
                "cot_redundancy": [],
                "length_of_memory": len(descs) if descs is not None else 0,
                "is_format_error": False,
                "has_invalid_face": False,
                "over_thinking": False,
                "output_memory": descs if descs is not None else [],
                "descs": descs,
                "eval_descs": descs[:self.memory_length_threshold] if descs is not None else [],
                "spans": spans,
                "txt": txt,                
        })        

        if descs is None:
            info["group_type"] = "parse_error"
            info["is_format_error"] = True
            info["fail_parsing"] = 1
            return info
        if think_len > self.think_length_threshold:
            info["group_type"] = "think_length_error"
            info["fail_parsing"] = 1
            info["over_thinking"] = True
            return info

        eval_descs = info["eval_descs"]

        invalid_face_rids = set()
        for j, d in enumerate(descs):
            if self._invalid_face_tag(d):
                invalid_face_rids.add(f"{j+1}")

        info["has_face_list"] = [1 if FACE_PATTERN.search(d) else 0 for d in eval_descs]
        info["has_invalid_face"] = True if len(invalid_face_rids) > 0 else False

        correctness_key = "correctness"
        coh_key = "label"

        if correctness_key not in task_results or coh_key not in task_results:
            info["group_type"] = "reward_error"
            info["fail_parsing"] = 2
            return info

        corr_vals = self.get_safe_values(task_results, correctness_key, default_val=[0]*len(eval_descs))
        coh_vals = self.get_safe_values(task_results, coh_key, default_val=[0]*len(eval_descs))

        if len(corr_vals) != len(eval_descs) or len(coh_vals) != len(eval_descs):
            info["group_type"] = "reward_error"
            info["fail_parsing"] = 2
            return info

        corr_cot = self.get_safe_values(task_results, f"{correctness_key}_rationale", default_val=[""]*len(eval_descs))
        corr_cot += [""] * (len(descs) - len(corr_cot))

        cot_red = self.get_safe_values(task_results, f"{coh_key}_rationale", default_val=[""]*len(eval_descs))
        cot_red += [""] * (len(descs) - len(cot_red))

        if len(eval_descs) == 0:
            info["redundancy_list"].append(0)
            info["correctness_list"].append(0)
            info["memory_count"].append(0)
            info["num_correct"] = 0

        for j, desc in enumerate(descs):
            rid = j+1
            token_len = spans[j][1] - spans[j][0] if j < len(spans) else 0
            info["memory_token_list"].append(token_len)
            invalid_face = True if rid in invalid_face_rids else False
            info["invalid_face_list"].append(1 if invalid_face else 0)
            info["memory_count"].append(j)
            if j < len(eval_descs):
                is_valid = not (invalid_face or token_len > self.memory_token_threshold)
                is_correct = (float(corr_vals[j]) > 0.5 and is_valid)

                info["cot_correctness"].append(corr_cot[j])
                info["correctness_list"].append(float(corr_vals[j]) if is_correct else 0)
                info["ori_correctness_list"].append(corr_vals[j])
                info["cot_redundancy"].append(cot_red[j])
                info["redundancy_list"].append(coh_vals[j])
                info["valid_list"].append(float(is_correct and coh_vals[j]))
            else:
                info["correctness_list"].append(0)
                info["redundancy_list"].append(0)
                info["valid_list"].append(0)

        info["num_correct"] = sum(info["correctness_list"])
        info["num_wrong"] = max(0, len(descs) - info["num_correct"])
        info["group_type"] = "normal"

        return info            

    def compute_token_level_rewards(
        self,
        token_id_matrix,
        extra_info=None,
        timeout=480,
    ):
        extra_info = extra_info or {}

        preceding_description = extra_info.get("preceding_description", "")
        if not self.example_questions:
            raise RuntimeError(
                "question_path is required to run LLM-judge evaluation. "
                "Pass it via `+reward_model.reward_kwargs.question_path=/path/to/questions.txt`."
            )
        sample_size = min(5, len(self.example_questions))
        downstream_prompts = "\n- " + "\n- ".join(random.sample(self.example_questions, k=sample_size))
        decoded_info, group = self.build_item(token_id_matrix, extra_info)

        info = {
            "preceding_description": preceding_description if preceding_description is not None else "",
        }

        if group:
            task_results = self._evaluate_all_tasks(group, downstream_prompts, extra_info, timeout)
        else:
            task_results = {}

        info = self.calculate_item_reward(decoded_info, task_results, info)
        info = self._stage1_score(extra_info, info, downstream_prompts, task_results)

        all_rewards = info["final_score"]
        del info["descs"], info["spans"], info["ids"]

        return all_rewards, info

    def _usefulness_comparison(self, sample_info_list, extra_info, result_map,
                            downstream_prompts="", timeout=15):
        extra_info = extra_info or {}
        ref_descs = self._parse_ref_memories(extra_info.get("ref_outputs"))
        rep = sample_info_list if sample_info_list["fail_parsing"] in [0, 2] else None

        sample_info_list["ref_loss"] = -1
        sample_info_list["cur_usefulness_vs_ref"] = 0.0
        sample_info_list["ref_loss_second"] = -1
        sample_info_list["curr_win_final"] = -1

        if rep and ref_descs:
            rep_text = rep["descs"][0]
            ref_text = ref_descs[0]
            score_key = "more_useful"
            def ask(blocks):
                prompt = HELPFULNESS_EPISODIC_PROMPT.format(
                    example_questions=downstream_prompts,
                    blocks_text=json.dumps(blocks, ensure_ascii=False, indent=4),
                )
                r = self._process_task([{"type": "text", "text": prompt}], score_key, None, self.model_name_text, timeout)
                try:
                    x = r.get(score_key)
                    x = x.get(score_key)["1"]
                    return (None if x is None else str(x).strip()), r.get(f"{score_key}_responses", "")
                except:
                    return None, r.get(f"{score_key}_responses", "")

            a, raw_a = ask({"0": rep_text, "1": ref_text})   # A: 0=rep
            b, raw_b = ask({"0": ref_text, "1": rep_text})   # B: 1=rep
            rep_win_a = (a == "0")
            rep_loss_a = (a == "1")
            rep_win_b = (b == "1")
            rep_loss_b = (b == "0")
            bad_a = a in (None, "-1")
            bad_b = b in (None, "-1")

            rep["ref_loss"] = int(rep_win_a) if not bad_a else 2
            rep["ref_loss_second"] = int(rep_win_b) if not bad_b else 2
            rep["pair_choice_a"], rep["pair_choice_b"] = a, b

            if (rep_win_a and rep_win_b) or (rep_win_a and bad_b) or (rep_win_b and bad_a):
                sample_info_list["curr_win_final"] = 1
            elif (rep_loss_a and rep_loss_b) or (bad_a and rep_loss_b) or (bad_b and rep_loss_a):
                sample_info_list["curr_win_final"] = 0
            else:
                sample_info_list["curr_win_final"] = 2

            g = result_map.setdefault("global", {})
            g.update({
                "pair_choice_a": a, "pair_choice_b": b,
                "more_useful_responses_a": raw_a,
                "more_useful_responses_b": raw_b,
            })
        elif rep and not ref_descs:
            rep["ref_loss"] = 1
            rep["ref_loss_second"] = 1
        elif not rep and ref_descs:
            sample_info_list["ref_loss"] = 0
            sample_info_list["ref_loss_second"] = 0

        return sample_info_list

    def _evaluate_all_tasks(self, groups, downstream_prompts, extra_info, timeout=240):
        tasks = ["correctness", "label"]
        task_results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            future_to_type = {
                ex.submit(self._run_task, groups, task_type, downstream_prompts, timeout): task_type
                for task_type in tasks
            }

            for fut in as_completed(future_to_type):
                task_type = future_to_type[fut]
                try:
                    res = fut.result()
                    task_results[task_type] = res
                except Exception as e:
                    print(f"[EvalTask] {task_type} failed: {e}")
                    task_results[task_type] = {}  
        final_results = {"global":{}}
        for task_type, task_res in task_results.items():
            final_results.update(task_res.get(task_type, {}))
            final_results["global"].update({f"{task_type}_responses": task_res.get(f"{task_type}_responses", "")})
        return final_results


    def _run_task(self, groups, task_type, downstream_prompts, timeout=480):
        if task_type == "correctness":
            return self._eval_llm(groups, task_type, downstream_prompts, timeout)
        elif task_type == "label":
            return self._eval_redundancy_episodic(groups, task_type, downstream_prompts, timeout)

    def _eval_llm(self, groups, task_type, downstream_prompts, timeout=480):
        tmpl = self.prompts.get(task_type)
        if not tmpl:
            return {task_type: {}}

        descs = groups.get("descs", [])
        if not descs:
            return {task_type: {}, f"{task_type}_responses": ""}
        blocks_text = descs[0]

        prompt = tmpl.format(
                example_questions=downstream_prompts,
                preceding_json=groups.get("preceding_description", ""),
                blocks_text=blocks_text,
            )

        inputs = [{"type": "video", "video": groups["video_path"]}, {"type": "text", "text": prompt}]
        model_name = self.model_name_video

        results_g = self._process_task(inputs, task_type, None, model_name, timeout)
        results = results_g.get(task_type, {})
        responses = results_g.get(f"{task_type}_responses", "")
        format_results = {task_type: results, f"{task_type}_responses": responses}
        return format_results

    @timeout(600, default={}, raise_err=False)
    def _eval_redundancy_episodic(self, groups, task_type, downstream_prompts, timeout=480):
        tmpl = self.prompts.get(task_type)
        if not tmpl:
            return {task_type: {}}
        responses = {}
        descs = groups.get("descs", [])
        if descs and descs[0]:
            prompt = tmpl.format(
                example_questions=downstream_prompts,
                preceding_json=groups.get("preceding_description", ""),
                blocks_text=descs[0],
                )
            inputs = [{"type": "text", "text": prompt}]
            try:
                results_g = self._process_task(inputs, task_type, None, self.model_name_text, timeout)
                results = results_g.get(task_type, {})
                responses = results_g.get(f"{task_type}_responses", "")
            except Exception as e:
                print(f"[EvalLLM] {task_type} failed: {e}")
                results = {f"{task_type}_rationale": {}, f"{task_type}": {}}
        else:
            results = {f"{task_type}_rationale": {}, f"{task_type}": {}}
        return {task_type: results, f"{task_type}_responses": json.dumps(responses, indent=2)}

    def _process_task(self, task_inputs, task_type, expected_idx, model_name, timeout):
        last_responses = ""
        for attempt in range(1, 20):
            try:
                messages = generate_messages(task_inputs)
                responses = get_response_with_retry(model_name, messages, timeout)
                last_responses = responses
                llm_responses = self._parse_llm_response(responses, expected_idx, task_type)
                return {task_type: llm_responses, f"{task_type}_responses": responses}
            except Exception as e:
                print(f"[EvalLLM] {task_type} failed: {e} (attempt {attempt})")
                last_error = e
        print(f"[EvalLLM] {task_type} failed: {last_error}")
        return {task_type: {}, f"{task_type}_responses": last_responses}      
    def _validate_score_value(self, v):
        if isinstance(v, bool):
            return True
        if isinstance(v, (int, float)):
            return True
        if isinstance(v, str):
            try:
                float(v.strip())
                return True
            except Exception:
                return False
        return False
    def _parse_llm_response(self, responses, expected_idx, task_type):
        parsed = loads(repair_json((responses.strip().strip("`json").strip())))
        score_key = f"{task_type}"

        if not isinstance(parsed, dict):
            raise ValueError("Missing or invalid JSON object.")
        
        if expected_idx is None:
            if not self._validate_score_value(parsed[score_key]):
                raise ValueError(f"not int value: {parsed}")
            parsed_format = {score_key: {"1": parsed[score_key]}, f"{score_key}_rationale": {"1": parsed.get(f"{score_key}_rationale", "")}}
            return parsed_format
        else:
            idx_dict = parsed.get(score_key)
            if set(idx_dict.keys()) != expected_idx:
                raise ValueError(f"idx mismatch. expected={expected_idx}, got={set(idx_dict.keys())}")
            for k, v in idx_dict.items():
                if not self._validate_score_value(v):
                    raise ValueError(f"invalid {score_key}[{k}] value type: {type(v)} {str(v)[:80]}")
            return parsed

    def _parse_ref_memories(self, ref_outputs):
        import numpy
        if isinstance(ref_outputs, list) or isinstance(ref_outputs, numpy.ndarray):
            return [str(x).strip() for x in ref_outputs if str(x).strip()]
        
        if isinstance(ref_outputs, str):
            s = ref_outputs.strip()
            if not s:
                return None
            return [s]
        return None

    def _stage1_score(
        self,
        extra_info: dict,
        info: dict,
        downstream_prompts: str,
        result_map: dict,
    ):
        # Eval-only metrics; reward_score is unused during DPO training (pairs come from data).
        # On parse / reward errors, all metrics are set to 0.
        if info["fail_parsing"] in [1, 2]:
            info["acc"] = 0.0
            info["redundancy"] = 0.0
            info["total_rate"] = 0.0
        else:
            n = max(1, len(info.get("descs") or []))
            info["acc"] = sum(info.get("correctness_list") or []) / n
            info["redundancy"] = sum(info.get("redundancy_list") or []) / n
            info["total_rate"] = sum(info.get("valid_list") or []) / n

        info["final_score"] = 0.0
        info["global_results"] = result_map.get("global", {})
        info["downstream_prompts"] = downstream_prompts

        self._usefulness_comparison(info, extra_info, result_map, downstream_prompts)

        return info

@register("m3_agent")
class M3AgentRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
        """
        super().__init__(config, tokenizer, compute_score)
        reward_kwargs = config.reward.get("reward_kwargs", {})

        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.reward_evaluator = MultiRewardEvaluator(tokenizer=self.tokenizer, **reward_kwargs)


    async def run_single(self, data) -> torch.Tensor | dict[str, Any]:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        batch = data_item.batch
        non_tensor = data_item.non_tensor_batch
        token_id_matrix = batch["responses"]
        root_path = non_tensor.get("root_path", "./outputs")
        global_step = str(non_tensor.get("global_steps", 0))
        file_dir = os.path.join(root_path, global_step)
        os.makedirs(file_dir, exist_ok=True)
        types = non_tensor["type"]
        input_blocks = non_tensor["input"]
        ref_outputs = non_tensor["response"]
        video_ids = non_tensor["id"]
        if "reward" in non_tensor:
            reward_score = float(non_tensor["reward"])
            return {
            "reward_score": reward_score,
            "reward_extra_info": {
                "reward_log": {},  
            },
        }

        batch["uid"] = torch.zeros((1,), device=token_id_matrix.device, dtype=torch.long)
    
        extra_info = self._extract_extra_info(input_blocks)
        extra_info["ref_outputs"] = ref_outputs
        
        reward_score, reward_log = self.reward_evaluator.compute_token_level_rewards(
            token_id_matrix,
            extra_info=extra_info,
        )

        safe_vid = str(video_ids).replace("/", "-slash-", 1)
        file_path = os.path.join(file_dir, f"{safe_vid}.json")

        payload = {
            "video_id": safe_vid,
            "type": types,
            "global_step": global_step,
            "num_items": 1,
            "input": input_blocks,
            "output": ref_outputs,
            "reward": reward_log,
        }
        _atomic_json_dump(payload, file_path)

        return {
            "reward_score": reward_score,
            "reward_extra_info": {
                "reward_log": reward_log,  
            },
        }

    def _extract_extra_info(self, input_list):
        video_path = None
        text_items = [it["text"] for it in input_list if it.get("type") == "text" and "text" in it and it["text"] is not None]
        last_text = text_items[-1] if text_items else ""
        preceding = last_text.split("[Description of the preceding part]:")[1].split("\n\n- Generate subsequent descriptions not")[0].strip()
        preceding = preceding if isinstance(preceding, str) else ""
        for item in input_list:
            if item["type"] == "video":
                video_path = item["video"]
                break
        return {
            "video_path": video_path,
            "preceding_description": preceding,
        }