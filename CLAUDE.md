# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SurvLLM tunes `meta-llama/Llama-3.1-8B-Instruct` for survival analysis by extracting key clinical signals from long-form discharge summaries. The pipeline is a full HFRL/RLAIF chain: SFT → preference labeling → (DPO or RM + PPO) → vLLM inference. QLoRA (4-bit nf4) plus `peft` adapters is used throughout; FSDP-QLoRA is supported for multi-GPU.

The README and many in-code comments are in Korean — preserve that when editing.

## Common Commands

All training entrypoints accept a YAML config via TRL's `TrlParser`. Logs are conventionally redirected into `logs/` (so `nohup` + `&` is the norm, not `&&`).

### Data preparation
```bash
# CSV -> JSON dataset dispatcher: format inferred from columns.
#   has `assistant` column           -> SFT  (sft_train_dataset.json, sft_test_dataset.json, test_label_sft.csv)
#   has `chosen`/`rejected` columns  -> DPO  (dpo_train_dataset.json, dpo_test_dataset.json, test_label_dpo.csv)
#   --ppo=true                       -> PPO  (prompt-only)
#   neither                          -> inference JSON (same basename as --target)
python csv_to_json_dataset.py --target=data/<file>.csv --encoding=utf-8 --system=data/system_prompt.txt [--ppo=true] [--name_tag=...]
```

### Training (single GPU)
Config and Log versions are just specify example.
```bash
nohup python SFT.py --config config/SFT_config_v1.2.0.yaml   > logs/sft_log_v1.2.0.txt   &
nohup python RM.py  --config config/RM_config_v1.1.2.7.yaml  > logs/rm_log_v1.1.2.7.txt  &
nohup python PPO.py --config config/PPO_config_v1.1.2.10.yaml > logs/ppo_log_v1.1.2.10.txt &
nohup python DPO.py --config config/DPO_config_v1.1.2.1.yaml > logs/dpo_log_v1.1.2.1.txt &
```

### Training (multi-GPU, FSDP-QLoRA)
Edit `num_processes` in `config/fsdp_config_qlora.yaml` (or `fsdp_config_qlora_dpo.yaml` for DPO) to match GPU count.
```bash
accelerate launch --config_file config/fsdp_config_qlora.yaml \
  SFT.py --config config/SFT_config_multi_GPU.yaml

NCCL_TIMEOUT=600 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --config_file config/fsdp_config_qlora_dpo.yaml \
  DPO.py --config config/DPO_config_multi_GPU.yaml
```
Set `multi_gpu: true` in the script config. In distributed mode, `SaveInferenceResultsCallback` and `utils.excel_integrate` are skipped automatically — they only run on single-GPU.

### Quantized base model + inference (vLLM)
Inference always goes through vLLM, even with unmerged adapters (massive speedup; first compile is slow).
```bash
# 1) Save an nf4-quantized copy of the base model once (also embeds the SFT-adapter tokenizer)
python gen_llama_nf4.py

# 2) Inference with a LoRA adapter on top of the quantized base
python vllm_inference.py \
  --base_model_path=base_model/Llama-3.1-8B-Instruct-nf4 \
  --adapter_path=adapter/<adapter-folder> \
  --inference_data=data/<input>.json \
  --output_dir=<output>.csv \
  --gen_nums=1 --sampling=True --temperature=0.4 --repetition_penalty=1.0 \
  --gpu_memory_util=0.9 --seed=42
```

### AI Feedback (RLAIF) preference labeling
```bash
python preference_AIF.py \
  --model_name=Qwen/Qwen3-30B-A3B \
  --preference_name=data/generated_data_v1.2.0.csv \
  --discharge_name=data/dpo_prompt_data.csv
```

### W&B
`WANDB_MODE=offline` is hardcoded in every trainer to avoid zombie processes. Sync manually:
```bash
wandb sync --include-offline wandb/latest-run
```

`rlhf.sh` and `ppo.sh` are reference cheat-sheets of the above; they are **not** runnable end-to-end (see the comment at the top of `rlhf.sh`).

## Architecture

### Pipeline
```
SFT (4-bit QLoRA on Llama-3.1-8B-Instruct)
  └── adapter/Zip-Llama-sft-vX
        ├──> vllm_inference.py generates N candidates per prompt
        │      └── preference_AIF.py (Qwen3-30B judge -> JSON ranks) -> DPO pairs
        │
        ├── DPO (loads SFT adapter twice as `policy` + `reference`)
        │     └── adapter/Zip-Llama-aligned-vX   (policy subfolder)
        │
        └── RM (sequence classifier on same base) -> PPO (policy = SFT adapter, reward = RM adapter)
              └── adapter/PPO-Llama-vX
```

### Adapter directory convention (critical for inference)
`vllm_inference.py` decides where to load the LoRA from by string-matching the adapter path:
- if the path **contains `sft`** → load that path directly
- otherwise → load `<adapter_path>/policy`

So an adapter folder must contain `sft` in its name OR have a `policy/` subdirectory. DPO/PPO outputs follow the latter (the policy adapter is saved under `policy/`, and DPO additionally has `reference/` for the frozen SFT adapter).

### TrlParser + YAML configs
Each script defines one or more `@dataclass`es (`ScriptArguments`, plus `LoraArguments` for SFT/RM) that TRL's `TrlParser` merges with the trainer's own config class (`SFTConfig`, `DPOConfig`, `RewardConfig`, experimental `PPOConfig`). All hyperparameters live in `config/*.yaml`; do not hardcode them in Python. SFT/RM filter `LoraArguments` against `LoraConfig.__init__.__code__.co_varnames` before instantiating `LoraConfig` — this is intentional so the YAML can carry extra metadata fields without breaking `LoraConfig`.

### Chat template
SFT installs a custom Llama-3 chat template with explicit `{% generation %}` markers so `assistant_only_loss: true` masks user/system tokens from the loss. `gen_llama_nf4.py` copies the tokenizer from an SFT adapter into `base_model/Llama-3.1-8B-Instruct-nf4` so vLLM picks up the same template downstream.

### `utils/` (single-GPU only)
- `SaveInferenceResultsCallback` — runs inference on `test_ds` at every save step and writes per-step results.
- `excel_integrate(run_name)` — collates those per-step files into one workbook at the end of training.
- `remove_hangul` — utility used by `csv_to_json_dataset.py`.

Both `SaveInferenceResultsCallback` and `excel_integrate` are skipped when `multi_gpu: true` because FSDP shard semantics break them.

### Dataset shape
After `csv_to_json_dataset.py`, training JSON files are conversational (`messages` field with `system`/`user`/`assistant` roles). DPO files additionally have `chosen`/`rejected`. The system prompt is read from a separate text file (default `data/system_prompt.txt`) so it can be edited without touching code. The dataset-flow timeline is documented in `data/README.md`.

## Project-specific Gotchas

- **Do not merge a QLoRA adapter back into a 4-bit base model.** Re-quantizing after merge mangles the SFT model. Keep adapters separate; vLLM loads them at runtime via `LoRARequest`.
- **DPO uses "adapter twice loaded"**: the SFT adapter is loaded as both the trainable `policy` and the frozen `reference`. Configs set `model_adapter_name: policy` / `ref_adapter_name: reference`. Multi-GPU DPO works but uses noticeably more memory.
- **Sequence lengths are large** (`max_length: 16384`, sometimes 32k for vLLM). Per the README, expect ≥100GB VRAM + ≥200GB system RAM for the documented config.
- **flash-attention-2 is required** (`attn_implementation="flash_attention_2"`). Ubuntu 20.04 needs a GLibc bump; 22.04+ recommended.
- **Adapter folder naming matters** — see "Adapter directory convention" above. Renaming an SFT adapter to drop `sft` will silently route inference to a non-existent `policy/` subdir.
- **`preference_AIF.py` lets the judge reason freely, then extracts a rank JSON** (`{"1": rank, ..., "5": rank}`). No structured outputs: the judge may emit `<think>...</think>` reasoning (`max_tokens=512`); the last valid JSON object after `</think>` is taken as the verdict. Outputs whose reasoning was truncated (`<think>` without `</think>`) are discarded to avoid mistaking the system prompt's example JSON for a real ranking. The evaluation system prompt lives in `preference_system_prompt.txt` (same for every request → vLLM prefix caching); the user prompt template is embedded in the script. Subject_ids whose 5 generations are all identical are filtered out before inference (no preference signal possible). A label is valid if it is a dict over `{"1".."5"}` with integer values in 1–5 and not all equal (partial ties are allowed and resolve to the earliest position). Failed outputs (truncated reasoning or invalid JSON) are retried up to `--max_retries` times (default 2), each round with seed+1 and doubled generation budget; only subjects that fail every round are dropped. With valid labels, the lowest rank (best) becomes `chosen` and the highest (worst) becomes `rejected`.
