"""nf4(bitsandbytes) 양자화 베이스 모델에 QLoRA 어댑터를 CPU에서 병합하여 HF 체크포인트로 저장.

핵심 원리
- nf4 가중치를 bnb의 `dequantize_4bit`로 역양자화하면, 학습/추론 시 커널이 실제로 사용하는 bf16 가중치와
  완전히 동일하므로 (양자화 노이즈까지 그대로) 이 단계는 무손실이다.
- LoRA delta(= alpha/r * B @ A)는 fp32로 계산하여 더한다. 단, 최종 저장 dtype이 bf16이면 delta(~1e-5)가
  bf16 가중치의 반올림 단위(half-ulp ~3e-5)보다 작아 대부분 사라진다. 무손실 병합이 필요하면 `--dtype fp32`.
- `--dequant_only`를 주면 어댑터 없이 역양자화된 베이스만 저장한다 (vLLM에서 bnb 경로 대신 bf16 + LoRA로 사용 가능).

사용 예 (프로젝트 루트에서, GPU 불필요):
    CUDA_VISIBLE_DEVICES="" python utils/merge_qlora_adapter.py \
        --nf4_base base_model/Llama-3.1-8B-Instruct-nf4 \
        --adapter adapter/PPO-Llama-v1.1.2.15H \
        --output merged_model/PPO-Llama-v1.1.2.15H-fp32 --dtype fp32
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from bitsandbytes.functional import QuantState, dequantize_4bit
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

QUANT_STATE_SUFFIXES: tuple[str, ...] = (
    "absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state.bitsandbytes__nf4",
)
QUANT_MARKER = ".quant_state.bitsandbytes__nf4"
LORA_PREFIX = "base_model.model."
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
UNSUPPORTED_ADAPTER_FIELDS = ("rank_pattern", "alpha_pattern", "modules_to_save")
DTYPES: dict[str, torch.dtype] = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DTYPE_NAMES: dict[torch.dtype, str] = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
DEFAULT_SHARD_BYTES = 5 * 1024**3
TOKENIZER_FILES: tuple[str, ...] = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
TMP_SHARD_PREFIX = "tmp-shard-"


@dataclass(frozen=True)
class LoraAdapter:
    """메모리에 올린 LoRA 어댑터 (fp32 텐서) + 스케일 계수."""

    tensors: dict[str, torch.Tensor]
    scaling: float

    @classmethod
    def load(cls, adapter_dir: Path) -> "LoraAdapter":
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"adapter_config.json 없음: {adapter_dir} (policy/ 하위 규약을 확인하세요)")
        config = json.loads(config_path.read_text())
        if config.get("use_dora"):
            raise ValueError("DoRA 어댑터는 단순 B@A 병합이 불가능합니다.")
        unsupported = [f for f in UNSUPPORTED_ADAPTER_FIELDS if config.get(f)]
        if unsupported:
            raise ValueError(f"지원하지 않는 어댑터 설정: {unsupported} (모듈별 rank/alpha, modules_to_save 미지원)")
        rank, alpha = config["r"], config["lora_alpha"]
        scaling = alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank
        with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as fh:
            tensors = {k: fh.get_tensor(k).float() for k in fh.keys()}
        cls._validate_pairs(tensors)
        return cls(tensors=tensors, scaling=scaling)

    @staticmethod
    def _validate_pairs(tensors: dict[str, torch.Tensor]) -> None:
        """모든 키가 lora_A/lora_B 쌍이어야 한다 (embedding LoRA 등 병합 불가 텐서가 조용히 무시되지 않도록)."""
        unknown = [k for k in tensors if not k.endswith(LORA_SUFFIXES)]
        if unknown:
            raise ValueError(f"병합할 수 없는 어댑터 텐서: {unknown[:5]} ...")
        modules = {k.removesuffix(s) for k in tensors for s in LORA_SUFFIXES if k.endswith(s)}
        orphans = [m for m in modules if not all(m + s in tensors for s in LORA_SUFFIXES)]
        if orphans:
            raise ValueError(f"lora_A/lora_B 짝이 맞지 않는 모듈: {orphans[:5]} ...")

    def delta(self, weight_key: str) -> torch.Tensor | None:
        """`model.layers.N.xxx.weight`에 대응하는 scaling * B @ A (fp32). 대상이 아니면 None."""
        prefix = LORA_PREFIX + weight_key.removesuffix(".weight")
        lora_a, lora_b = self.tensors.get(f"{prefix}.lora_A.weight"), self.tensors.get(f"{prefix}.lora_B.weight")
        if lora_a is None or lora_b is None:
            return None
        return self.scaling * (lora_b @ lora_a)

    def targeted_keys(self) -> set[str]:
        return {k.removesuffix(".lora_A.weight").removeprefix(LORA_PREFIX) + ".weight"
                for k in self.tensors if k.endswith(".lora_A.weight")}


class ShardReader:
    """safetensors 샤드 인덱스를 따라 텐서를 읽는다 (파일 핸들 lazy open, with 문으로 해제)."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.weight_map: dict[str, str] = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._handles: dict[str, object] = {}

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.model_dir / shard), "pt").__enter__()
        return self._handles[shard].get_tensor(key)

    def is_quantized(self, key: str) -> bool:
        return key + QUANT_MARKER in self.weight_map

    def weight_keys(self) -> list[str]:
        """quant_state 보조 텐서를 제외한 실제 파라미터 키 (샤드 순서 유지)."""
        return [k for k in self.weight_map if not any(k.endswith("." + s) for s in QUANT_STATE_SUFFIXES)]

    def dequantize(self, key: str) -> torch.Tensor:
        """double quant 미사용 베이스는 nested_* 키가 없으므로 존재하는 보조 텐서만 모은다."""
        qs_dict = {f"{key}.{s}": self.get(f"{key}.{s}") for s in QUANT_STATE_SUFFIXES if f"{key}.{s}" in self.weight_map}
        quant_state = QuantState.from_dict(qs_dict=qs_dict, device="cpu")
        return dequantize_4bit(self.get(key), quant_state)


def merge_tensor(weight: torch.Tensor, delta: torch.Tensor | None, dtype: torch.dtype, key: str) -> torch.Tensor:
    if delta is None:
        return weight.to(dtype).contiguous()
    if delta.shape != weight.shape:
        raise ValueError(f"{key}: LoRA delta shape {tuple(delta.shape)} != weight {tuple(weight.shape)}")
    merged = (weight.float() + delta).to(dtype).contiguous()
    if dtype == torch.float16 and not torch.isfinite(merged).all():
        raise ValueError(f"{key}: fp16 변환 시 overflow 발생")
    return merged


def iter_merged_tensors(
    reader: ShardReader, adapter: LoraAdapter | None, dtype: torch.dtype,
) -> Iterator[tuple[str, torch.Tensor]]:
    """역양자화 + (선택) LoRA 병합된 텐서를 하나씩 생성."""
    for key in reader.weight_keys():
        weight = reader.dequantize(key) if reader.is_quantized(key) else reader.get(key)
        delta = adapter.delta(key) if adapter is not None else None
        yield key, merge_tensor(weight, delta, dtype, key)


class ShardWriter:
    """텐서를 max_shard_bytes 단위 safetensors 샤드로 순차 기록 (메모리에는 현재 샤드만 유지)."""

    def __init__(self, out_dir: Path, max_shard_bytes: int) -> None:
        self.out_dir, self.max_shard_bytes = out_dir, max_shard_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.total_bytes = 0
        self.shard_keys: list[list[str]] = []

    def add(self, key: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self.pending[key] = tensor
        self.pending_bytes += nbytes
        self.total_bytes += nbytes

    def _flush(self) -> None:
        index = len(self.shard_keys)
        save_file(self.pending, str(self.out_dir / f"{TMP_SHARD_PREFIX}{index:05d}.safetensors"), metadata={"format": "pt"})
        self.shard_keys.append(list(self.pending))
        self.pending, self.pending_bytes = {}, 0

    def finish(self) -> dict[str, str]:
        """남은 샤드를 기록하고 임시 파일을 최종 이름으로 바꾼 뒤 weight_map 반환."""
        if self.pending:
            self._flush()
        if not self.shard_keys:
            raise ValueError("저장할 텐서가 없습니다.")
        total = len(self.shard_keys)
        weight_map: dict[str, str] = {}
        for index, keys in enumerate(self.shard_keys):
            name = "model.safetensors" if total == 1 else f"model-{index + 1:05d}-of-{total:05d}.safetensors"
            os.replace(self.out_dir / f"{TMP_SHARD_PREFIX}{index:05d}.safetensors", self.out_dir / name)
            weight_map.update({k: name for k in keys})
        return weight_map

    def cleanup_tmp(self) -> None:
        for tmp in self.out_dir.glob(f"{TMP_SHARD_PREFIX}*.safetensors"):
            tmp.unlink()


def save_sharded(tensors: Iterator[tuple[str, torch.Tensor]], out_dir: Path, max_shard_bytes: int) -> tuple[dict[str, str], int]:
    """(weight_map, 텐서 바이트 총합). 실패 시 임시 샤드를 정리한다."""
    writer = ShardWriter(out_dir, max_shard_bytes)
    try:
        for key, tensor in tensors:
            writer.add(key, tensor)
        return writer.finish(), writer.total_bytes
    except BaseException:
        writer.cleanup_tmp()
        raise


def write_index(weight_map: dict[str, str], total_bytes: int, out_dir: Path) -> None:
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def write_metadata(nf4_dir: Path, out_dir: Path, dtype: torch.dtype, tokenizer_dir: Path | None = None) -> None:
    """config.json에서 quantization_config를 제거하고 dtype을 갱신, 토크나이저는 한 곳에서만 통째로 복사."""
    config = json.loads((nf4_dir / "config.json").read_text())
    config.pop("quantization_config", None)
    config["dtype"] = config["torch_dtype"] = DTYPE_NAMES[dtype]
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    if (nf4_dir / "generation_config.json").exists():
        shutil.copy2(nf4_dir / "generation_config.json", out_dir / "generation_config.json")
    source = tokenizer_dir if tokenizer_dir is not None and (tokenizer_dir / "tokenizer.json").exists() else nf4_dir
    for name in TOKENIZER_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, out_dir / name)
    logger.info("tokenizer files copied from %s", source)


def resolve_adapter_dir(adapter_path: Path) -> Path:
    """vllm_inference.py와 동일 규칙: 경로 문자열에 'sft'가 있으면 그대로, 아니면 policy/ 하위."""
    return adapter_path if "sft" in str(adapter_path) else adapter_path / "policy"


def prepare_output_dir(nf4_dir: Path, out_dir: Path) -> None:
    """베이스 덮어쓰기와 이전 실행의 샤드 잔재(샤드 수가 바뀌면 섞임)를 막는다."""
    if out_dir.resolve() == nf4_dir.resolve():
        raise ValueError("--output 이 --nf4_base 와 같습니다. 베이스를 덮어쓸 수 없습니다.")
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = list(out_dir.glob("*.safetensors"))
    if stale:
        raise FileExistsError(f"{out_dir} 에 safetensors 파일이 이미 있습니다 ({len(stale)}개). 비우고 다시 실행하세요.")


def merge(nf4_dir: Path, adapter_path: Path | None, out_dir: Path, dtype: torch.dtype,
          max_shard_bytes: int = DEFAULT_SHARD_BYTES) -> None:
    prepare_output_dir(nf4_dir, out_dir)
    adapter = LoraAdapter.load(resolve_adapter_dir(adapter_path)) if adapter_path else None
    with ShardReader(nf4_dir) as reader:
        if adapter is not None:
            missing = adapter.targeted_keys() - set(reader.weight_keys())
            if missing:
                raise ValueError(f"어댑터 대상 모듈이 베이스에 없음: {sorted(missing)[:5]} ...")
            logger.info("adapter scaling=%.4f, targeted weights=%d", adapter.scaling, len(adapter.targeted_keys()))
        weight_map, total_bytes = save_sharded(iter_merged_tensors(reader, adapter, dtype), out_dir, max_shard_bytes)
    write_index(weight_map, total_bytes, out_dir)
    write_metadata(nf4_dir, out_dir, dtype, tokenizer_dir=adapter_path)
    logger.info("saved %d tensors (%s, %.2f GB) -> %s", len(weight_map), DTYPE_NAMES[dtype], total_bytes / 1024**3, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nf4_base", type=Path, default=Path("base_model/Llama-3.1-8B-Instruct-nf4"))
    parser.add_argument("--adapter", type=Path, default=None, help="어댑터 폴더 (policy/ 규약은 자동 처리)")
    parser.add_argument("--output", type=Path, required=True, help="저장 폴더 (예: merged_model/<name>), 비어 있어야 함")
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--dequant_only", action="store_true", help="어댑터 없이 역양자화된 베이스만 저장")
    parser.add_argument("--max_shard_gb", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.dequant_only and args.adapter is None:
        raise SystemExit("--adapter 를 지정하거나 --dequant_only 를 사용하세요.")
    merge(args.nf4_base, None if args.dequant_only else args.adapter, args.output,
          DTYPES[args.dtype], int(args.max_shard_gb * 1024**3))


if __name__ == "__main__":
    main()
