## nohup python vllm_inference.py --gpu_memory_util=0.45 &

import os

## vLLM 왜이래... 되긴 됐는데...
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["VLLM_USE_V1"] = "0" 
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import pandas as pd

## apply chat template
def template_dataset(example):
    return {"prompt": tokenizer.apply_chat_template(example["messages"], tokenize = False, add_generation_prompt = True)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type = str, default = "base_model/Llama-3.1-8B-Instruct-nf4", help = "base model (nf4 양자화 후 저장, 개조된 토크나이저 포함) path")
    parser.add_argument("--adapter_path", type = str, default = "adapter/Zip-Llama-aligned", help = "adapter config path")
    parser.add_argument("--inference_data", type = str, default = "data/inference_data.json", help = "추론 프롬프트 파일 위치 (json type). messages column에 프롬프트가 전부 정리되어 있어야 합니다.")
    parser.add_argument("--gen_nums", type = int, default = 1, help = "프롬프트 당 생성 횟수")
    parser.add_argument("--output_dir", type = str, default = "data/inference_result.csv", help = "생성 결과 파일 저장 위치")
    parser.add_argument("--gpu_memory_util", type = float, default = 0.5)
    parser.add_argument("--sampling", type = bool, default = True, help = "샘플링 여부")
    parser.add_argument("--temperature", type = float, default = 0.4, help = "sampling temperature")
    parser.add_argument("--repetition_penalty", type = float, default = 1.0, help = "반복적 생성에 의한 패널티. 1.0 초과하여 설정 시 적용됨")
    parser.add_argument("--seed", type = int, default = None, help = "시드 설정")

    args = parser.parse_args()

    base_model_path = args.base_model_path
    adapter_path = args.adapter_path

    llm = LLM(
        model= base_model_path,
        dtype=torch.bfloat16,
        trust_remote_code = True,
        max_model_len = 32768,
        gpu_memory_utilization = args.gpu_memory_util,
        enable_lora = True,
        max_lora_rank = 64
    )

    if args.sampling:
        sampling_params = SamplingParams(
            n = args.gen_nums,
            temperature = args.temperature,
            top_p = 0.9,
            max_tokens = 512,
            seed = args.seed,
            repetition_penalty = args.repetition_penalty
        )

    else:
        sampling_params = SamplingParams(
            n = args.gen_nums,
            temperature = 0.0,
            max_tokens = 512,
            seed = args.seed,
            repetition_penalty = args.repetition_penalty
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast = True)
    inference_data = load_dataset("json", data_files = args.inference_data, split = "train")
    inference_data = inference_data.map(template_dataset, remove_columns = ["messages"])
    prompts = inference_data["prompt"]

    outputs = llm.generate(
        prompts, 
        sampling_params,
        lora_request = LoRARequest("adapter", 1, adapter_path if "sft" in adapter_path else os.path.join(adapter_path, "policy"))
    )

    data = []
    idx = inference_data["subject_id"]

    for i, output in enumerate(outputs):
        current_subject_id = idx[i]
        
        for j in range(args.gen_nums):
            row = {
                "subject_id": current_subject_id,
                "generated_text": output.outputs[j].text.strip()
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(args.output_dir, index=False, encoding="utf-8-sig")