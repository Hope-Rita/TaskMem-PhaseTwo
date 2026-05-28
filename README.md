# TaskMem — Phase Two Training

This repository contains the Phase Two training pipeline of TaskMem,
distributed as a fork of [verl](https://github.com/volcengine/verl) based on
upstream commit `32705dc1`.

## Installation

```bash
git clone --recurse-submodules https://github.com/Hope-Rita/TaskMem-PhaseTwo.git
cd TaskMem-PhaseTwo
bash bootstrap.sh
```

`bootstrap.sh` registers the bundled vLLM adapter plugins
(`qwen{2,3}vllm_ada`) via `pip install -e .` and installs two additional
runtime dependencies (`json_repair`, `httpx==0.23.3`).

The bundled `verl/` directory provides this fork's modified verl package; a
separate verl checkout is therefore not required. Verl's runtime
dependencies (vLLM, Ray, PyTorch, Hydra, Transformers, FSDP, etc.) must be
provisioned independently; refer to the
[upstream verl installation guide](https://github.com/volcengine/verl) for
details.

All commands documented below assume the working directory is the
repository root so that the bundled `verl/` package takes precedence on
`sys.path`.

## Environment Variables

The following variables are required for both training and validation:

```bash
export VLLM_PLUGINS=qwen3vllm_ada
export VLLM_MODELS=qwen3vllm_ada
export VLLM_USE_V1=1
```

`VLLM_PLUGINS` activates the adapter-aware vLLM model class. `VLLM_MODELS`
declares which parameter group is trainable; the value `qwen3vllm_ada` marks
the adapter parameters as trainable while keeping the base model frozen.

The following variable is optional and restricts adapter training to a
specified set of decoder layer indices:

```bash
export VLLM_ADA_TRAIN_LAYERS=22,23,24
```

The following variable is required only when validation metrics are
computed:

```bash
export M3_AGENT_API_CONFIG=configs/api_config.json
```

## Configuration Matrix

| Resource                          | Training | Validation |
| --------------------------------- | :------: | :--------: |
| Parquet data and base HF model    |    ✓     |     ✓      |
| `VLLM_PLUGINS`, `VLLM_MODELS`     |    ✓     |     ✓      |
| `M3_AGENT_API_CONFIG`             |          |     ✓      |
| `reward_kwargs.question_path`     |          |     ✓      |

During training the LLM judge is not invoked, as preference pairs are
supplied directly by the dataset. The judge is engaged only when validation
is enabled, in which case both the API configuration file and the question
file must be provided.

### LLM Judge API Configuration

The judge configuration is a JSON document located at the path specified by
`M3_AGENT_API_CONFIG`:

```json
{
  "gpt-4o-2024-11-20":  { "azure_endpoint": "...", "api_version": "...", "api_key": "..." },
  "gemini-2.5-flash":   { "azure_endpoint": "...", "api_version": "...", "api_key": "..." }
}
```

Each value may also be supplied as a list of dictionaries, in which case the
configured endpoints are used in round-robin fashion.

### Question File

A plain-text file containing one question per line. Its path is provided
through the `+reward_model.reward_kwargs.question_path` override at launch
time.

## Pipeline

The pipeline consists of four stages. The following shell variables are
referenced throughout:

| Variable      | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| `$BASE_MODEL` | Path to the HuggingFace directory containing adapter weights  |
| `$TRAIN_PQ`   | Path to the DPO preference parquet                            |
| `$VAL_PQ`     | Path to the validation parquet                                |
| `$QUESTIONS`  | Path to the question file (validation only)                   |
| `$EXP_NAME`   | Experiment name                                               |
| `$CKPT_ROOT`  | Output root; per-step `steer.pt` artefacts are written here   |

### Stage 1 — Training

```bash
bash run.sh \
    data.train_files=$TRAIN_PQ \
    data.val_files=$VAL_PQ \
    actor_rollout_ref.model.path=$BASE_MODEL \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=$CKPT_ROOT \
    trainer.total_training_steps=10 \
    trainer.nnodes=4 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=dpo \
    actor_rollout_ref.actor.optim.lr=1e-3 \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    +reward_model.reward_kwargs.memory_length_threshold=5 \
    +reward_model.reward_kwargs.memory_token_threshold=200 \
    +reward_model.reward_kwargs.think_length_threshold=4800
```

### Stage 2 — Steer Injection

```bash
python add_steer.py \
    --model_path $BASE_MODEL \
    --steer_file $CKPT_ROOT/10/steer.pt \
    --save_path  $CKPT_ROOT/10/huggingface-steer \
    --extra_files_path $BASE_MODEL
```

### Stage 3 — Layer-wise Steer Scaling

```bash
python change_values_in_steer.py \
    --src_path  $CKPT_ROOT/10/huggingface-steer \
    --layer_idx 22 \
    --scale     1.5 \
    --extra_files_path $BASE_MODEL
```

The resulting model is written to
`$CKPT_ROOT/10/huggingface-steer-layer22-scale1p5`.

### Stage 4 — Evaluation

```bash
bash run.sh \
    data.val_files=$VAL_PQ \
    actor_rollout_ref.model.path=$CKPT_ROOT/10/huggingface-steer-layer22-scale1p5 \
    trainer.experiment_name=${EXP_NAME}_eval \
    trainer.default_local_dir=${CKPT_ROOT}_eval \
    trainer.nnodes=1 \
    trainer.val_only=True \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    +reward_model.reward_kwargs.question_path=$QUESTIONS \
    +reward_model.reward_kwargs.memory_length_threshold=5 \
    +reward_model.reward_kwargs.memory_token_threshold=200 \
    +reward_model.reward_kwargs.think_length_threshold=4800
```

## Reward Configuration

Default values are defined in `verl/trainer/config/reward/reward.yaml`. The
accepted keys correspond to the constructor parameters of
`MultiRewardEvaluator.__init__`:

| Key                       | Default               | Description                                                  |
| ------------------------- | --------------------- | ------------------------------------------------------------ |
| `max_response_length`     | `8196`                | Maximum response length scanned for `<think>` blocks         |
| `think_length_threshold`  | `1600`                | Reported as a metric; no penalty is applied                  |
| `memory_length_threshold` | `5`                   | Only the first `N` memory descriptions are evaluated         |
| `memory_token_threshold`  | `25`                  | Per-memory token cap; longer entries are marked invalid      |
| `model_name_text`         | `gpt-4o-2024-11-20`   | Text judge LLM (validation only)                             |
| `model_name_video`        | `gemini-2.5-flash`    | Video judge LLM (validation only)                            |
| `question_path`           | `null`                | Required when validation is enabled                          |

The reward score is fixed at `0.0`, since DPO operates on preference pairs
supplied by the dataset. The metrics `acc`, `redundancy`, and `total_rate`
are reported for evaluation purposes only. The primary tracked metric is
`trajectory/test_time/final_win_ratio`, the pairwise win rate against the
reference policy.
