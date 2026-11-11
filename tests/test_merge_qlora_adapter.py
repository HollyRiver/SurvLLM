"""utils/merge_qlora_adapter.py 단위 테스트 (CPU, bitsandbytes 필요).

실행: CUDA_VISIBLE_DEVICES="" python tests/test_merge_qlora_adapter.py   (pytest가 있으면 pytest로도 실행 가능)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from bitsandbytes.functional import dequantize_4bit, quantize_4bit
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.merge_qlora_adapter import LoraAdapter, ShardReader, merge, resolve_adapter_dir  # noqa: E402

OUT, IN, RANK, ALPHA = 64, 128, 4, 2
Q_KEY, NORM_KEY = "model.layers.0.self_attn.q_proj.weight", "model.layers.0.input_layernorm.weight"


def _quantize(weight: torch.Tensor, double_quant: bool) -> tuple[torch.Tensor, object]:
    return quantize_4bit(weight, blocksize=64, compress_statistics=double_quant, quant_type="nf4")


def _make_nf4_base(base_dir: Path, weight: torch.Tensor, norm: torch.Tensor, double_quant: bool) -> None:
    packed, qs = _quantize(weight, double_quant)
    tensors = {Q_KEY: packed, NORM_KEY: norm}
    tensors.update({f"{Q_KEY}.{k}": v for k, v in qs.as_dict(packed=True).items()})
    save_file(tensors, str(base_dir / "model.safetensors"), metadata={"format": "pt"})
    index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in tensors}}
    (base_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    config = {"model_type": "llama", "dtype": "bfloat16", "quantization_config": {"quant_method": "bitsandbytes"}}
    (base_dir / "config.json").write_text(json.dumps(config))
    (base_dir / "tokenizer_config.json").write_text("{}")


def _make_adapter(adapter_dir: Path, lora_a: torch.Tensor, lora_b: torch.Tensor, extra_config: dict | None = None) -> None:
    policy = adapter_dir / "policy"
    policy.mkdir(parents=True)
    prefix = "base_model.model." + Q_KEY.removesuffix(".weight")
    save_file({f"{prefix}.lora_A.weight": lora_a, f"{prefix}.lora_B.weight": lora_b},
              str(policy / "adapter_model.safetensors"), metadata={"format": "pt"})
    config = {"r": RANK, "lora_alpha": ALPHA, "peft_type": "LORA", **(extra_config or {})}
    (policy / "adapter_config.json").write_text(json.dumps(config))


def _fixture(tmp: Path, double_quant: bool = True, extra_config: dict | None = None) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    weight = (torch.randn(OUT, IN) * 0.02).to(torch.bfloat16)
    norm = torch.ones(OUT, dtype=torch.bfloat16)
    lora_a, lora_b = torch.randn(RANK, IN) * 1e-2, torch.randn(OUT, RANK) * 1e-3
    base_dir, adapter_dir = tmp / "nf4", tmp / "PPO-test"
    base_dir.mkdir()
    _make_nf4_base(base_dir, weight, norm, double_quant)
    _make_adapter(adapter_dir, lora_a, lora_b, extra_config)
    expected_w = dequantize_4bit(*_quantize(weight, double_quant))
    return base_dir, adapter_dir, expected_w, (ALPHA / RANK) * (lora_b @ lora_a)


def _read_all(out_dir: Path) -> dict[str, torch.Tensor]:
    weight_map = json.loads((out_dir / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for shard in set(weight_map.values()):
        with safe_open(str(out_dir / shard), "pt") as fh:
            result.update({k: fh.get_tensor(k) for k in fh.keys()})
    return result


def _raises(fn, exc_type: type[BaseException]) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"{exc_type.__name__} 가 발생해야 함")


def test_dequantize_matches_bitsandbytes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir, _, expected_w, _ = _fixture(Path(tmp))
        with ShardReader(base_dir) as reader:
            assert reader.is_quantized(Q_KEY) and not reader.is_quantized(NORM_KEY)
            assert torch.equal(reader.dequantize(Q_KEY), expected_w)


def test_dequantize_without_double_quant() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir, _, expected_w, _ = _fixture(Path(tmp), double_quant=False)
        with ShardReader(base_dir) as reader:
            assert f"{Q_KEY}.nested_absmax" not in reader.weight_map
            assert torch.equal(reader.dequantize(Q_KEY), expected_w)


def test_fp32_merge_is_exact_and_strips_quant_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir, adapter_dir, expected_w, delta = _fixture(Path(tmp))
        out_dir = Path(tmp) / "out"
        merge(base_dir, adapter_dir, out_dir, torch.float32, max_shard_bytes=4096)
        tensors = _read_all(out_dir)
        assert set(tensors) == {Q_KEY, NORM_KEY}, "quant_state 보조 텐서가 섞여 들어감"
        assert tensors[Q_KEY].dtype == torch.float32
        assert torch.allclose(tensors[Q_KEY], expected_w.float() + delta, atol=0, rtol=1e-6)
        assert torch.equal(tensors[NORM_KEY], torch.ones(OUT))
        assert len(list(out_dir.glob("model-*.safetensors"))) == 2 and not list(out_dir.glob("tmp-shard-*"))
        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        assert index["metadata"]["total_size"] == OUT * IN * 4 + OUT * 4
        config = json.loads((out_dir / "config.json").read_text())
        assert "quantization_config" not in config and config["dtype"] == "float32"
        assert (out_dir / "tokenizer_config.json").exists()


def test_bf16_merge_rounds_like_peft() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir, adapter_dir, expected_w, delta = _fixture(Path(tmp))
        out_dir = Path(tmp) / "out"
        merge(base_dir, adapter_dir, out_dir, torch.bfloat16)
        merged = _read_all(out_dir)[Q_KEY]
        assert merged.dtype == torch.bfloat16
        assert torch.equal(merged, (expected_w.float() + delta).to(torch.bfloat16))


def test_dequant_only_keeps_base_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir, _, expected_w, _ = _fixture(Path(tmp))
        out_dir = Path(tmp) / "out"
        merge(base_dir, None, out_dir, torch.bfloat16)
        assert torch.equal(_read_all(out_dir)[Q_KEY], expected_w)


def test_adapter_delta_none_for_untargeted_key() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, adapter_dir, _, delta = _fixture(Path(tmp))
        adapter = LoraAdapter.load(adapter_dir / "policy")
        assert adapter.delta(NORM_KEY) is None
        assert torch.allclose(adapter.delta(Q_KEY), delta)
        assert adapter.targeted_keys() == {Q_KEY}


def test_rejects_unsupported_adapter_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, adapter_dir, _, _ = _fixture(Path(tmp), extra_config={"alpha_pattern": {"q_proj": 32}})
        _raises(lambda: LoraAdapter.load(adapter_dir / "policy"), ValueError)


def test_refuses_to_overwrite_base_or_nonempty_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir, adapter_dir, _, _ = _fixture(Path(tmp))
        _raises(lambda: merge(base_dir, adapter_dir, base_dir, torch.float32), ValueError)
        out_dir = Path(tmp) / "out"
        merge(base_dir, adapter_dir, out_dir, torch.float32)
        _raises(lambda: merge(base_dir, adapter_dir, out_dir, torch.float32), FileExistsError)


def test_resolve_adapter_dir_matches_vllm_rule() -> None:
    assert resolve_adapter_dir(Path("adapter/Zip-Llama-sft-v1.1.2")) == Path("adapter/Zip-Llama-sft-v1.1.2")
    assert resolve_adapter_dir(Path("adapter/PPO-Llama-v1.1.2.15H")) == Path("adapter/PPO-Llama-v1.1.2.15H/policy")
    assert resolve_adapter_dir(Path("sft_runs/PPO-x")) == Path("sft_runs/PPO-x")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
