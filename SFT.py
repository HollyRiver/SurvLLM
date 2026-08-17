## ↓ run
## nohup python SFT.py --config config/SFT_config.yaml &

## imports
from datasets import load_dataset
from dataclasses import dataclass, field, fields    ## For TrlParser

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    set_seed
)
from trl import SFTTrainer, SFTConfig, TrlParser
from peft import LoraConfig

import logging
import torch

import os
import glob
import json
import random
import numpy as np
import pandas as pd
import wandb
import utils

## Zombie Process 발생 방지
os.environ["WANDB_MODE"] = "offline"    ## 수동 업데이트: wandb sync --include-offline ./wandb/latest-run
wandb.init(project = "RLHF")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

## TrlParser에 들어갈 class들을 커스터마이징: 하이퍼파라미터 저장
@dataclass  ## 데이터 보관 클래스를 간단하게 구축 가능: __init__, __repr__, __eq()__등의 메소드 자동 생성
class ScriptArguments:
    dataset_path: str = field(default = None, metadata = {"help": "dataset directory"})
    model_name: str = field(default = None, metadata = {"help": "사용할 모델 ID"})
    multi_gpu: bool = field(default = False, metadata = {"help": "Multi-GPU 사용 여부"})
    name_tag: str = field(default = '', metadata = {"help": "sft_***_dataset{tag}.json 형식으로 입력됨"})

@dataclass
class LoraArguments:
    r: int = field(default = 64, metadata = {"help": "update matrix의 rank. 작을수록 많이 압축하여 품질 저하됨, 메모리 많이 할당됨"})
    lora_alpha: int = field(default = 32, metadata = {"help": "∆Weight scaling factor. lora_alpha / r로 스케일링되며, 학습률 조정. 보통 1/2 수준으로 설정"})
    lora_dropout: float = field(default = 0.05, metadata = {"help": "update matrics에서 dropout 적용 확률"})
    bias: str = field(default = "none", metadata = {"help": "update matrix에 bias를 학습할 것인지 선택"})
    task_type: str = field(default = "CAUSAL_LM", metadata = {"help": "학습할 모형이 무엇인지 지정"})
    target_modules: list[str] = field(default = None, metadata = {"help": "학습에 반영할 모듈 설정"})


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
def main(script_args, training_args, lora_kwargs):
    ## loading dataset
    train_ds = load_dataset("json", data_files = os.path.join(script_args.dataset_path, f"sft_train_dataset{script_args.name_tag}.json"), split = "train")
    test_ds = load_dataset("json", data_files = os.path.join(script_args.dataset_path, f"sft_test_dataset{script_args.name_tag}.json"), split = "train")

    print(f"training dataset size: {train_ds.num_rows}\ntest dataset size: {test_ds.num_rows}")

    ## 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        use_fast = True,            ## Rust로 구현된 Fast Tokenizer 사용 (Qwen, RoPE, ChatGLM 등의 특이한 구조에서는 호환 안됨)
        trust_remote_code = True)   ## 모델 코드 전체 다운로드 후 사용
    tokenizer.pad_token = tokenizer.eos_token       ## 패딩할 토큰 설정
    tokenizer.padding_side = "left"                 ## 디코더이므로 왼쪽을 패딩 (마지막 토큰을 보고 생성)

    ## 데이터셋에 적합한 chat template 적용: generation 부분을 추가하여 assistant_only_loss 진행
    ## 모든 텍스트로 손실을 계산하고자 한다면 tokenizer에 기본으로 할당된 chat template로 충분
    ## jinja2 template engine 구문. 파이썬 문법과 거의 동일
    LLAMA_3_CHAT_TEMPLATE = (
        "{{ bos_token }}"
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + eos_token }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] +  eos_token }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n'}}"
                "{% generation %}"
                "{{ message['content'] +  eos_token }}"
                "{% endgeneration %}"
            "{% endif %}"
        "{% endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{%- endif %}"
    )

    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    ## 템플릿 적용사항 확인 (메인 프로세스에서만 해당 작업 수행, multi-GPU 사용 시 필요)
    with training_args.main_process_first():
        print("======== Log a few random samples from the processed training set ========")
        for index in random.sample(range(len(train_ds)), 2):
            print(tokenizer.apply_chat_template(train_ds[index]["messages"], tokenize = False))

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
        low_cpu_mem_usage = True,
        attn_implementation = "flash_attention_2",  ## flash_attention 연산 사용. sdpa가 더 빠르고 효율적일 수도 있음.
        quantization_config = bnb_config,
        dtype = torch.bfloat16                      ## 가중치 로드 데이터 타입. Llama-3.1-8B의 자료형으로 설정
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(**lora_kwargs)

    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = test_ds,
        processing_class = tokenizer,
        peft_config = peft_config
    )

    if training_args.assistant_only_loss:
        print("======== Log a first sample from the processed training set ========")
        print(f"masking area: {next(iter(trainer.train_dataset))["assistant_masks"][:100]} ...")

    ## 학습이 중단된 경우 이어서 진행할 수 있도록 설정
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    if not script_args.multi_gpu:
        inference_callback = utils.SaveInferenceResultsCallback(trainer=trainer, test_dataset=test_ds, model_name=training_args.output_dir.split("/")[-1])
        trainer.add_callback(inference_callback)

    trainer.train(resume_from_checkpoint = checkpoint)

    ## (분산 GPU 사용 시) 중간 체크포인트는 분할되어 저장, 훈련 종료 후 전체 상태 딕셔너리로 저장
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model()


if __name__ == "__main__":
    os.makedirs("wandb", exist_ok = True)
    # initial_folders = set(next(os.walk("wandb"))[1])

    parser = TrlParser((ScriptArguments, SFTConfig, LoraArguments))         ## 따로 저장된 파라미터 파싱
    script_args, training_args, lora_args = parser.parse_args_and_config()

    ## Lora Config에 유효한 입력값만 받을 수 있도록 커스터마이징. 원래 TrlParser에는 LoraConfig를 넣지 못함
    valid_keys = LoraConfig.__init__.__code__.co_varnames
    lora_kwargs = {
        f.name: getattr(lora_args, f.name)
        for f in fields(lora_args)
        if f.name in valid_keys
    }

    # seeding(training_args.seed)

    main(script_args, training_args, lora_kwargs)

    print("========== 학습 종료 ==========")

    ## ========== 추론 파일 종합 ===========
    if not script_args.multi_gpu:
        utils.excel_integrate(training_args.output_dir.split("/")[-1])

    ## ========== wandb 업로드 ==========
    # current_folders = set(next(os.walk("wandb"))[1])
    # new_folders = current_folders - initial_folders
    # os.system(f"wandb sync --include-offline ./wandb/{list(current_folders)[0]}")
    os.system(f"wandb sync --include-offline wandb/latest-run")