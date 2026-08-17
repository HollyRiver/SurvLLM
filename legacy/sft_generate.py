## nohup python sft_generate.py --adapter_name="adapter/Zip-Llama-sft" --output_name="gen_data.csv" --gen_nums=1 --temp=0.4 &

## SFT에서 온전한 모델을 픽스하고, 해당 어뎁터를 삽입
## temperature 설정은 1.0 정도로 해야 다양한 결과 나옴
# nohup python sft_generate.py --adapter_name="adapter/Zip-Llama-sft"\
#                              --output_name="gen_data.csv"\
#                              --gen_nums=5\
#                              --temp=1.0 &

import os
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_name", type = str, default = None, help = "STF model path")
    parser.add_argument("--output_name", type = str, default = "for_dpo_5_gen.csv")
    parser.add_argument("--gen_nums", type = int, default = 5)
    parser.add_argument("--temp", type = float, default = 0.7)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    config = PeftConfig.from_pretrained(args.adapter_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config = bnb_config,
        use_cache = True,
        dtype = torch.bfloat16,
        device_map = "cuda:0"
    )

    model = PeftModel.from_pretrained(model, args.adapter_name)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_name, use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    gen_ds = load_dataset("json", data_files = "data/dpo_resource.json")["train"]
    output_path = os.path.join("data", f"dpo_dataset_generated.csv")

    results = []

    model.eval()
    model = torch.compile(model)
    
    with torch.inference_mode():
        for idx in tqdm(range(gen_ds.num_rows)):
            ith_inference = {"subject_id" : gen_ds[idx]["subject_id"]}
            ith_inference["text"] = gen_ds[idx]["messages"][1]["content"]

            for i in range(args.gen_nums):
                input_ids = tokenizer.apply_chat_template(
                                gen_ds[idx]["messages"],
                                add_generation_prompt=True,
                                return_tensors="pt"
                ).to(model.device)

                terminators = [tokenizer.eos_token_id]

                outputs = model.generate(
                    input_ids,
                    max_new_tokens=1024,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=args.temp,
                    top_p = 0.95
                )

                response = outputs[0][input_ids.shape[-1]:]
                generation = tokenizer.decode(response, skip_special_tokens=True)
                ith_inference[f"Gen_{i}"] = generation

            results.append(ith_inference)

    pd.DataFrame(results).to_csv(f"data/{args.output_name}", encoding = "utf-8-sig")