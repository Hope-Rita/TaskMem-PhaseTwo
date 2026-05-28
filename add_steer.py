#!/usr/bin/env python3
import torch
import os
import shutil
import argparse
from transformers import AutoConfig, AutoModelForImageTextToText

from modeling.qwen3hf_ada import register

# Example usage:
#   python add_steer.py \
#       --model_path /path/to/hf_model \
#       --steer_file /path/to/steer.pt \
#       --save_path /path/to/output_dir \
#       --extra_files_path /path/to/base_hf_model

def parse_args():
    parser = argparse.ArgumentParser(description="Replace steer vectors in HF model")
    parser.add_argument("--model_path", type=str, required=True, help="Original HuggingFace model directory")
    parser.add_argument("--steer_file", type=str, required=True, help="Path to steer.pt file with 'matrix'")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the new model")
    parser.add_argument("--extra_files_path", type=str, default=None,
                        help="Optional path to copy extra tokenizer/config files from")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"], help="Torch dtype for model")
    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def main():
    args = parse_args()
    model_path = args.model_path
    steer_file = args.steer_file
    save_path = args.save_path
    torch_dtype = get_torch_dtype(args.torch_dtype)
    extra_files_path = args.extra_files_path

    register() 

    print(f"[INFO] Loading config from {model_path} ...")
    cfg = AutoConfig.from_pretrained(model_path)
    cfg.adapter_layers = "all"

    print(f"[INFO] Loading model from {model_path} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        config=cfg,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    print(f"[INFO] Loading steer from {steer_file} ...")
    steer_data = torch.load(steer_file)['matrix']  # (num_layers, hidden_dim)

    if steer_data.shape[0] != len(model.model.language_model.layers):
        raise ValueError(f"steer matrix layer count {steer_data.shape[0]} != model layers {len(model.model.language_model.layers)}")

    print("[INFO] Replacing steer vectors for all layers ...")
    for idx, layer in enumerate(model.model.language_model.layers):
        if not hasattr(layer, "adapter") or layer.adapter is None:
            raise RuntimeError(f"layer {idx} has no adapter")
        if not hasattr(layer.adapter, "steer"):
            raise RuntimeError(f"layer {idx}.adapter has no steer")
        layer.adapter.steer.data.copy_(steer_data[idx].to(layer.adapter.steer.device))

    os.makedirs(save_path, exist_ok=True)
    print(f"[INFO] Saving new model to {save_path} ...")
    model.save_pretrained(save_path, safe_serialization=True, max_shard_size="5GB")

    if extra_files_path is not None:
        extra_files = [
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
            "special_tokens_map.json",
            "processor_config.json",
            "preprocessor.json",
        ]
        print(f"[INFO] Copying extra files from {extra_files_path} ...")
        for fname in extra_files:
            src_file = os.path.join(extra_files_path, fname)
            dst_file = os.path.join(save_path, fname)
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)

    print(f"[INFO] Done. New model saved to {save_path}")


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()