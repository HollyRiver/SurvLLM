"""병합 모델 vs 미병합(nf4 + LoRA) 비교 — CPU 전용, GPU 불필요.

reference(ref_bf16): 역양자화 bf16 베이스 + PEFT LoRA(fp32) 미병합. 학습/vLLM의 bnb 경로가 실제로 계산하는 것과 수치적으로 동일.
variants:
  ref_fp32     : 같은 가중치/LoRA를 fp32로 연산 (연산 정밀도 차이의 바닥값)
  merged_fp32  : utils/merge_qlora_adapter.py --dtype fp32
  merged_fp16  : --dtype fp16 (가중치만 fp16 반올림, 연산은 fp32로 수행)
  merged_bf16  : --dtype bf16 (사용자가 기존에 시도한 방식과 동일한 반올림)
  no_lora_bf16 : 역양자화 베이스만 (LoRA 효과 크기의 기준점)
  orig_bf16_merge : HF 원본 bf16 Llama + LoRA를 PEFT merge_and_unload (노트북 마지막 셀 방식)
metrics: gold 응답 토큰 teacher-forcing KL(ref||var), top-1 일치율, gold NLL, greedy 생성문.

실행: CUDA_VISIBLE_DEVICES="" python debug/compare_merged_vs_lora.py --samples 8 7 9 --gen_samples 8 7
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
NF4_DIR = ROOT / "base_model/Llama-3.1-8B-Instruct-nf4"
DEQUANT_DIR = ROOT / "merged_model/Llama-3.1-8B-Instruct-nf4-dequant-bf16"
ADAPTER_DIR = ROOT / "adapter/PPO-Llama-v1.1.2.15H/policy"
MERGED = {name: ROOT / f"merged_model/PPO-Llama-v1.1.2.15H-{name}" for name in ("fp32", "fp16", "bf16")}
ORIG_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EOS_IDS = [128001, 128008, 128009]


def load_samples(path: Path, indices: list[int]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [rows[i] for i in indices]


def encode(tokenizer: PreTrainedTokenizerBase, messages: list[dict]) -> tuple[torch.Tensor, int]:
    """(prompt+gold 응답 전체 id, prompt 길이)."""
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, return_tensors="pt")
    full = tokenizer.apply_chat_template(messages, return_tensors="pt")
    assert torch.equal(full[0, : prompt.shape[1]], prompt[0]), "prompt가 full의 접두사가 아님"
    return full, prompt.shape[1]


@torch.inference_mode()
def response_logprobs(model: PreTrainedModel, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """gold 응답 토큰 위치의 log-softmax (fp32), shape (resp_len, vocab)."""
    logits = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids)).logits[0]
    return torch.log_softmax(logits[prompt_len - 1 : -1].float(), dim=-1)


def compare(ref_lp: torch.Tensor, var_lp: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    kl = (ref_lp.exp() * (ref_lp - var_lp)).sum(-1)
    return {
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "top1_agree": (ref_lp.argmax(-1) == var_lp.argmax(-1)).float().mean().item(),
        "gold_nll": -var_lp.gather(1, targets[:, None]).mean().item(),
    }


@torch.inference_mode()
def greedy(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt_ids: torch.Tensor, max_new: int) -> str:
    out = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), max_new_tokens=max_new,
                         do_sample=False, eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[-1])
    return tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)


def plain(path: Path | str, dtype: torch.dtype) -> Callable[[], PreTrainedModel]:
    return lambda: AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cpu", local_files_only=True)


def with_lora(path: Path, dtype: torch.dtype) -> Callable[[], PreTrainedModel]:
    def build() -> PreTrainedModel:
        model = PeftModel.from_pretrained(plain(path, dtype)(), ADAPTER_DIR)
        lora_dtype = next(p.dtype for n, p in model.named_parameters() if "lora_A" in n)
        logger.info("LoRA weight dtype = %s (미병합 경로)", lora_dtype)
        return model.eval()
    return build


def orig_bf16_merge() -> PreTrainedModel:
    """노트북 마지막 셀 방식: 원본 bf16 Llama + LoRA → merge_and_unload (bf16 반올림 + 양자화 불일치)."""
    return PeftModel.from_pretrained(plain(ORIG_MODEL_ID, torch.bfloat16)(), ADAPTER_DIR).merge_and_unload().eval()


def variant_builders() -> dict[str, Callable[[], PreTrainedModel]]:
    return {
        "ref_bf16": with_lora(DEQUANT_DIR, torch.bfloat16),
        "ref_fp32": with_lora(DEQUANT_DIR, torch.float32),
        "merged_fp32": plain(MERGED["fp32"], torch.float32),
        "merged_fp16": plain(MERGED["fp16"], torch.float32),
        "merged_bf16": plain(MERGED["bf16"], torch.bfloat16),
        "no_lora_bf16": plain(DEQUANT_DIR, torch.bfloat16),
        "orig_bf16_merge": orig_bf16_merge,
    }


def evaluate_variant(name: str, build: Callable[[], PreTrainedModel], tokenizer: PreTrainedTokenizerBase,
                     encoded: list[tuple[torch.Tensor, int]], gen_idx: list[int], max_new: int) -> dict:
    start = time.time()
    model = build().eval()
    logger.info("[%s] loaded in %.0fs", name, time.time() - start)
    logprobs = [response_logprobs(model, full, plen) for full, plen in encoded]
    logger.info("[%s] teacher-forced %d samples (%.0fs)", name, len(encoded), time.time() - start)
    gens = {i: greedy(model, tokenizer, encoded[i][0][:, : encoded[i][1]], max_new) for i in gen_idx}
    logger.info("[%s] greedy done (%.0fs total)", name, time.time() - start)
    del model
    gc.collect()
    return {"logprobs": logprobs, "gens": gens}


def summarize(results: dict[str, dict], encoded: list[tuple[torch.Tensor, int]], ref_name: str) -> dict:
    ref = results[ref_name]
    summary: dict[str, dict] = {}
    for name, res in results.items():
        per_sample = [compare(ref["logprobs"][i], res["logprobs"][i], full[0, plen:])
                      for i, (full, plen) in enumerate(encoded)]
        summary[name] = {k: sum(s[k] for s in per_sample) / len(per_sample) for k in per_sample[0]}
    return summary


def print_report(summary: dict, results: dict, gen_idx: list[int], subject_ids: list[int]) -> None:
    print("\n| variant | KL(ref||var) mean | KL max | top-1 agree | gold NLL |")
    print("|---|---|---|---|---|")
    for name, m in summary.items():
        print(f"| {name} | {m['kl_mean']:.5f} | {m['kl_max']:.4f} | {m['top1_agree']:.3f} | {m['gold_nll']:.4f} |")
    for i in gen_idx:
        print(f"\n=== greedy generation, sample idx {i} (subject_id {subject_ids[i]}) ===")
        for name, res in results.items():
            print(f"--- {name} ---\n{res['gens'][i]}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/sft_test_dataset.json")
    parser.add_argument("--samples", type=int, nargs="+", default=[8, 7, 9], help="teacher-forcing에 쓸 행 인덱스")
    parser.add_argument("--gen_samples", type=int, nargs="+", default=[0, 1], help="--samples 내 위치 (greedy 생성)")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--variants", nargs="+", default=list(variant_builders()),
                        help="평가할 변형 이름. 첫 번째 항목이 reference가 되므로 ref_bf16을 맨 앞에 둘 것")
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--out", type=Path, default=None, help="요약 JSON 저장 경로")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(NF4_DIR)
    samples = load_samples(args.data, args.samples)
    encoded = [encode(tokenizer, s["messages"]) for s in samples]
    logger.info("samples: %s", [(s["subject_id"], full.shape[1], plen) for s, (full, plen) in zip(samples, encoded)])
    builders = variant_builders()
    results = {name: evaluate_variant(name, builders[name], tokenizer, encoded, args.gen_samples, args.max_new_tokens)
               for name in args.variants}
    summary = summarize(results, encoded, ref_name=args.variants[0])
    print_report(summary, results, args.gen_samples, [s["subject_id"] for s in samples])
    if args.out:
        gens = {name: {str(i): g for i, g in res["gens"].items()} for name, res in results.items()}
        args.out.write_text(json.dumps({"summary": summary, "generations": gens}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
