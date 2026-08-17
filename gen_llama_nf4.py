from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    save_directory = "base_model/Llama-3.1-8B-Instruct-nf4"
    adapter_name = "adapter/Zip-Llama-aligned"

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