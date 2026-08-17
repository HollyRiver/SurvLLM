from pathlib import Path

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch

## utils/ 안으로 이동한 뒤에도 실행 위치와 무관하게 동작하도록 프로젝트 루트 기준으로 경로를 고정
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    save_directory = str(PROJECT_ROOT / "base_model" / "Llama-3.1-8B-Instruct-nf4")
    adapter_name = str(PROJECT_ROOT / "adapter" / "Zip-Llama-aligned")

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        quantization_config = bnb_config,
        use_cache = True,
        dtype = torch.bfloat16,
        device_map = "cuda:0"
    )

    model.save_pretrained(save_directory)

    tokenizer = AutoTokenizer.from_pretrained(adapter_name)
    tokenizer.save_pretrained(save_directory)