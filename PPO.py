## ↓ run
## nohup python PPO.py --config config/PPO_config.yaml & > dpo_config.out

## imports
from datasets import load_dataset
from trl import TrlParser
from trl.experimental.ppo import PPOTrainer, PPOConfig
from transformers import (
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoTokenizer, BitsAndBytesConfig, set_seed
)
import torch

from peft import PeftModel

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
    rm_name: str = field(default = None, metadata = {"help": "RM adapter"})
    multi_gpu: bool = field(default = False, metadata = {"help": "Multi-GPU 사용 여부"})
    max_length: int = field(default = 8192, metadata = {"help": "최대 토큰 사이즈"})


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
    ## loading dataset; eval_dataset 없으니까 오류나는데?
    train_ds = load_dataset("json", data_files = os.path.join(script_args.dataset_path, "ppo_train_dataset.json"), split = "train")
    test_ds = load_dataset("json", data_files = os.path.join(script_args.dataset_path, "ppo_test_dataset.json"), split = "train")

    ## 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        use_fast = True,
        trust_remote_code = True
    )
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

    def prepare_ppo_dataset(examples):
        prompt_strings = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in examples["messages"]
        ]

        tokens = tokenizer(
            prompt_strings,
            truncation=True,
            max_length=script_args.max_length,
            padding=False
        )

        return tokens

    train_ds = train_ds.map(prepare_ppo_dataset, batched=True, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(prepare_ppo_dataset, batched=True, remove_columns=test_ds.column_names)

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
        use_cache = False,                          ## VRAM 캐시 미사용, 추론 속도 저하. gradient_checkpointing과 동시 사용 불가
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
        is_trainable = True,
        adapter_name="policy",
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        device_map = None if script_args.multi_gpu else "cuda:0",
        use_cache = False,
        low_cpu_mem_usage = True,
        attn_implementation = "flash_attention_2",
        trust_remote_code = True,
        quantization_config = bnb_config,
        dtype = torch.bfloat16
    )

    ref_model.load_adapter(
        script_args.adapter_name,
        is_trainable = False,
        # adapter_name = "reference"
    )

    # model.set_adapter("policy")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # training_args.model_adapter_name = "policy"
    # training_args.ref_adapter_name = "reference"

    ## Reward Model 로드
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels = 1,
        device_map = None if script_args.multi_gpu else "cuda:0",
        use_cache = False,
        low_cpu_mem_usage = True,  
        attn_implementation = "flash_attention_2",
        trust_remote_code = True,
        quantization_config = bnb_config,
        dtype = torch.bfloat16
    )

    value_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels = 1,
        device_map = None if script_args.multi_gpu else "cuda:0",
        use_cache = False,
        low_cpu_mem_usage = True,  
        attn_implementation = "flash_attention_2",
        trust_remote_code = True,
        quantization_config = bnb_config,
        dtype = torch.bfloat16
    )

    reward_model = PeftModel.from_pretrained(
        reward_model,
        script_args.rm_name,
        is_trainable = False
    )

    value_model = PeftModel.from_pretrained(
        value_model,
        script_args.rm_name,
        is_trainable = True
    )

    value_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    value_model.enable_input_require_grads()

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    reward_model.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.pad_token_id = tokenizer.pad_token_id

    trainer = PPOTrainer(
        model = model,
        ref_model = ref_model,
        reward_model = reward_model,
        value_model = value_model,
        args = training_args,
        train_dataset= train_ds,
        eval_dataset = test_ds,
        processing_class = tokenizer
    )

    ## Monkey Patching
    def custom_gc_disable():
        if hasattr(trainer.model, "policy"):
            trainer.model.policy.gradient_checkpointing_disable()
        if hasattr(trainer.model, "value_model"):
            trainer.model.value_model.gradient_checkpointing_disable()

    def custom_gc_enable(*args, **kwargs):
        if hasattr(trainer.model, "policy"):
            trainer.model.policy.gradient_checkpointing_enable(*args, **kwargs)
        if hasattr(trainer.model, "value_model"):
            trainer.model.value_model.gradient_checkpointing_enable(*args, **kwargs)

    trainer.model.gradient_checkpointing_disable = custom_gc_disable
    trainer.model.gradient_checkpointing_enable = custom_gc_enable

    # 훈련에 참여하는 파라미터가 0이 아닌지 확인
    print("=== Policy Model Trainable Parameters ===")
    trainer.model.policy.print_trainable_parameters()
    
    print("=== Value Model Trainable Parameters ===")
    trainer.model.value_model.print_trainable_parameters()

    trainer.train()

    ## 최종 가중치 기준 생성 테이블 1회 추가 기록
    ## (생성 트리거가 update 1에 고정되어 마지막 update에서는 생성이 발생하지 않는 문제 보완)
    trainer.generate_completions(sampling = True)

    ## (분산 GPU 사용 시) 중간 체크포인트는 분할되어 저장, 훈련 종료 후 전체 상태 딕셔너리로 저장
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, PPOConfig))         ## 따로 저장된 파라미터 파싱
    script_args, training_args = parser.parse_args_and_config()

    # seeding(training_args.seed)

    main(script_args, training_args)

    print("========== 학습 종료 ==========")

    ## ========== wandb 업로드 ==========
    os.system(f"wandb sync --include-offline wandb/latest-run")