## ↓ run
## nohup python DPO.py --config config/DPO_config.yaml > dpo_config.out &

## imports
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer, TrlParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import torch

from peft import PeftModel, prepare_model_for_kbit_training

from dataclasses import dataclass, field
import argparse

import os
import json
import wandb
import random
import numpy as np

## Zombie Process 발생 방지
os.environ["WANDB_MODE"] = "offline"    ## 수동 업데이트: wandb sync --include-offline ./wandb/offline-*
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
wandb.init(project = "RLHF")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

## TrlParser에 들어갈 class들을 커스터마이징: 하이퍼파라미터 저장
@dataclass  ## 데이터 보관 클래스를 간단하게 구축 가능: __init__, __repr__, __eq()__등의 메소드 자동 생성
class ScriptArguments:
    dataset_path: str = field(default = None, metadata = {"help": "dataset directory"})
    model_name: str = field(default = None, metadata = {"help": "사용할 모델 ID"})
    adapter_name: str = field(default = None, metadata = {"help": "SFT 완료된 어뎁터"})
    multi_gpu: bool = field(default = False, metadata = {"help": "Multi-GPU 사용 여부"})
    name_tag: str = field(default = '', metadata = {"help": "dpo_***_dataset{tag}.json 형식으로 입력됨"})


def timer(func):
    """
    함수 실행 시간 출력
    """
    import time
    import datetime

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        sec = end - start
        worktime = str(datetime.timedelta(seconds=sec)).split(".")[0]
        print(f"Working Time: {worktime}")
        return result

    return wrapper


def seeding(seed):
    """
    시드 설정으로 인해 성능이 저하될 수 있음. dataloader worker에도 시드 설정이 필요할 수 있음
    """
    set_seed(seed)

    torch.manual_seed(seed)                 ## cpu seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        ## gpu seed
        torch.cuda.manual_seed_all(seed)    ## multi-gpu seed

    torch.backends.cudnn.deterministic = True   ## nondeterministic algorithm을 사용하지 않도록 설정
    torch.backends.cudnn.benchmark = False      ## cuDNN의 여러 convolution algorithm들을 실행하고, 벤치마킹하여 가장 빠른 알고리즘 사용: 안함.

    np.random.seed(seed)
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)    ## hash 알고리즘 관련
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   ## oneDNN 옵션 해제. 수치 연산 순서 고정 (성능 저하, 속도 저하)

@timer
def main(script_args, training_args):
    ## loading dataset
    train_ds = load_dataset("json", data_files = os.path.join(script_args.dataset_path, f"dpo_train_dataset{script_args.name_tag}.json"), split = "train")
    test_ds = load_dataset("json", data_files = os.path.join(script_args.dataset_path, f"dpo_test_dataset{script_args.name_tag}.json"), split = "train")

    ## 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        use_fast = True,            ## Rust로 구현된 Fast Tokenizer 사용 (Qwen, RoPE, ChatGLM 등의 특이한 구조에서는 호환 안됨)
        trust_remote_code = True)   ## 모델 코드 전체 다운로드 후 사용
    tokenizer.pad_token = tokenizer.eos_token       ## 패딩할 토큰 설정
    tokenizer.padding_side = "left"                 ## 디코더이므로 왼쪽을 패딩 (마지막 토큰을 보고 생성)

    ## jinja2 template engine chat template: masking 제외
    LLAMA_3_CHAT_TEMPLATE = (
        "{{ bos_token }}"
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + eos_token }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] +  eos_token }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n'}}"
                "{{ message['content'] +  eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{%- endif %}"
    )

    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    ## 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,                        ## 4비트 양자화
        bnb_4bit_use_double_quant = True,           ## 추가 양자화로 성능 손실 없이 파라미터당 0.4bit 추가 절약
        bnb_4bit_quant_type = "nf4",                ## 양자화 데이터 타입 지정: 4비트 기반 모델 훈련 시 사용
        bnb_4bit_compute_dtype = torch.bfloat16,    ## Llama-3.1-8B의 학습 자료형. 저장은 4비트지만, attention 연산은 해당 포맷으로 역양자화하여 처리
        bnb_4bit_quant_storage = torch.bfloat16 if script_args.multi_gpu else None
    )

    ## 모델 로드 및 설정
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        device_map = None if script_args.multi_gpu else "cuda:0",
        use_cache = False,                          ## VRAM 캐시 미사용, 추론 속도 저하. gradienc_checkpointing과 동시 사용 불가
        low_cpu_mem_usage = True,                   ## CPU RAM 사용량 적게 사용...
        attn_implementation = "flash_attention_2",  ## flash_attention 연산 사용. sdpa가 더 빠르고 효율적일 수도 있음.
        trust_remote_code = True,
        quantization_config = bnb_config,
        dtype = torch.bfloat16                      ## 가중치 로드 데이터 타입. Llama-3.1-8B의 자료형으로 설정
    )

    ## 어뎁터 부착
    model = PeftModel.from_pretrained(
        model,
        script_args.adapter_name,
        is_trainable=True,
        adapter_name="policy",
    )

    model.load_adapter(script_args.adapter_name, adapter_name = "reference")
    model.set_adapter("policy")

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if script_args.multi_gpu:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.config.use_cache = False
        
        print("\n[FSDP FINAL SOLUTION] Converting EVERYTHING to bfloat16 (Adapters + LayerNorms)...")
    
        # 1. 모든 파라미터 순회
        for _, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

        print("✅ All target parameters cast to bfloat16.")

    training_args.model_adapter_name = "policy"
    training_args.ref_adapter_name = "reference"

    trainer = DPOTrainer(
        model,
        args = training_args,
        train_dataset= train_ds,
        eval_dataset = test_ds,
        processing_class = tokenizer
    )

    ## 학습이 중단된 경우 이어서 진행할 수 있도록 설정
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint = checkpoint)

    ## (분산 GPU 사용 시) 중간 체크포인트는 분할되어 저장, 훈련 종료 후 전체 상태 딕셔너리로 저장
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig))         ## 따로 저장된 파라미터 파싱
    script_args, training_args = parser.parse_args_and_config()

    # seeding(training_args.seed)

    main(script_args, training_args)

    print("========== 학습 종료 ==========")

    ## ========== wandb 업로드 ==========
    os.system(f"wandb sync --include-offline wandb/latest-run")