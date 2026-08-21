# LLM 사후 학습을 통한 텍스트 정형화 페이즈

&nbsp;LLM을 이용한 생존 분석 투 페이즈 파이프라인의 첫 번째 페이즈입니다. 경량 오픈소스 LLM인 [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 모델을 사후 학습하여, 장문의 퇴원요약지 텍스트에서 규격화된 핵심 분석들을 문장의 형태로 추출하는 파이프라인입니다. 기본적인 SFT + Alignment 프로세스에 [QLoRA](https://arxiv.org/abs/2305.14314), [Load to adapter twice](https://huggingface.co/docs/trl/v0.8.1/en/dpo_trainer#using-option-3---load-the-adapter-twice)를 적용하여 학습을 진행했습니다.

## 1. Full Pipeline

![Two Pipeline Overview](https://github.com/HollyRiver/SurvLLM/blob/main/Fig/%5BFig1%5D%20Full%20Pipeline.png?raw=true)

&nbsp;해당 연구의 목적은 임상 상황의 생존 분석에서 고전적으로 사용되는 공변량을 넘어, 퇴원요약지라는 비정형 텍스트를 추가 정보로 활용하여 환자의 생존 시간을 더욱 효과적으로 예측하는 것입니다. 이를 위해서 단순히 퇴원요약지 텍스트와 복잡한 지시사항을 LLM에 입력하는 방법을 생각할 수 있습니다. 하지만, 날것의 퇴원요약지는 규격화되어있지 않고, 부가적인 정보와 입원 중 반복 측정에 따른 동적인 검사 수치들로 가득 채워져 있습니다. 따라서, end-to-end 모델 대신, 아래의 둘로 나뉘어진 파이프라인을 구성했습니다:

1. 퇴원요약지 텍스트를 약속된 형태와 내용으로 규격화하는 첫 번째 트랙
2. 규격화된 텍스트를 바탕으로 환자의 생존 시간을 예측하는 두 번째 트랙

## 2. First-Track Overview

&nbsp;본 리포지토리는 첫 번째 페이즈를 다룹니다. 구체적으로는 LLM이 텍스트 규격화 작업을 전문적으로 수행할 수 있도록 사후 학습을 수행하고, 사후 학습을 마친 모델을 vLLM 엔진에 올려 실제로 추론하기까지의 모든 코드를 기록합니다. 따라서 해당 프로세스는 생존 분석의 첫 번째 페이즈만으로서 활용될 수도 있지만, 약간의 시스템/유저 프롬프트와 하이퍼 파라미터 세팅 변경을 통해 일반적인 LLM 사후 학습 프로세스로 활용될 수 있습니다.

![4 Method Branches](https://github.com/HollyRiver/SurvLLM/blob/main/Fig/%5BFig2%5D%20First%20Track%20Training%20Branches.png?raw=true)

&nbsp;LLM을 사후 학습하는 방법은 다양합니다. 여기서는 총 네 가지 방법론으로 실험을 진행하여, 방법론 간 결과의 차이를 식별하기로 했습니다. 의료 전문가가 작성한 크기 100의 메시지 데이터셋 하나를 기반으로 SFT를 수행한 뒤, 선호도 데이터셋을 인간 또는 오픈소스 LLM의 레이블링으로 비교하여 인간 피드백(RLHF), 인공지능 피드백(RLAIF)으로의 분기가 나뉩니다. 여기에 두 가지 선호도 조정 알고리즘인 DPO와 PPO가 사용되어 총 네 가지 방법론이 구분됩니다. 각 방법론의 학습 순서는 아래 표와 동일하게 진행됩니다.

|Alignment 알고리즘|인간 피드백 선호도 레이블링|AI 피드백 선호도 레이블링|
|-:|:-:|:-:|
|**DPO**|SFT -> DPO + RLHF|SFT -> DPO + RLAIF|
|**PPO**|SFT -> RM + RLHF -> PPO|SFT -> RM + RLAIF -> PPO|

&nbsp;학습 과정에는 여러 과정이 필요하나, 각 방법론에서 산출되는 최종 모델은 하나입니다. 즉, 텍스트 정형화 태스크 하나는 end-to-end로 수행 가능합니다. 여기서는 완성된 모델을 양자화된 LLM과 LoRA 어댑터로 저장하므로, 애플리케이션으로 빌드될 때에는 가정용 GPU와 같은 저비용 환경에서도 구동할 수 있다는 장점이 있습니다.


## 3. Requirement and Setup

* 120GB 이상의 VRAM 및 128GB 이상의 시스템 메모리 권장. 처리 가능한 최대 토큰 시퀀스 길이(max context)를 더 길게 설정한다면, 이보다 많이 필요합니다.
* 본 연구에서는 매우 긴 문장까지 처리할 수 있도록 max context를 16,384로 설정했습니다. 학습 데이터셋의 토큰 분포를 확인한 후, 적절한 값으로 설정해주세요. 학습 데이터셋에서 토큰 길이가 매우 긴 문장의 수가 적다면, Truncation 대신 해당 샘플을 제거하고 max context를 줄이세요. 레이블이 있는 상황에서 Assistant Only Loss로 학습되므로, 해당 샘플이 학습에 주는 영향력은 없을 것입니다.
* VRAM이 부족하다면, 다음 순서대로 처리하는 것을 고려할 수 있습니다.
  1. 학습 데이터에서 토큰 길이가 매우 긴 이상치를 제거하고, max context를 감소시킴
  2. 학습 데이터 텍스트가 어느 정도 정형화되어있는 경우, 최종 요약 문장과 관련 없는 세션을 정규표현식 등을 사용하여 기계적으로 제거. 이후 max context를 감소시킴
  3. 더 작은 모델을 사용 (e.g., [Gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B), [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct))
  4. TRL 라이브러리 대신 Unsloth를 사용하여 최적화: (해당 라이브러리의 경우, 실무에서는 종종 사용되나 연구용으로는 잘 사용되지 않습니다. 8GB VRAM에서도 학습 가능한 세팅이 존재하니, 자원이 부족하다면 고려는 할 수 있습니다.)
* cuda 12.8, Ubuntu 22.04에서 구동시켰으나, Ubuntu 24.04 이상을 권장합니다. Ubuntu 22.04 버전에서는 flash-attention 실행을 위한 다운그레이드 및 GLibc 업데이트가 필요할 수 있습니다. [참고: \[ISSUE\] GLIBC_2.32 not found](https://github.com/modular/modular/issues/3684#issuecomment-2480409734)
* Dependencies (아래를 순차적으로 설치)
   * `transformers bitsandbytes datasets sentencepiece accelerate trl peft wandb openai pqdm`: `pip install`로 일괄 설치
   * `pytorch`: [\[Pytorch\] Get Started](https://pytorch.org/get-started/locally/)
   * `flash-attention`: [\[GitHub\] flash-attention](https://github.com/Dao-AILab/flash-attention), [\[Wheels\]](https://github.com/Dao-AILab/flash-attention/discussions/1838)
   * `vllm`: [Installation > GPU](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
* Docker Container 사용을 권장합니다.
  * 컨테이너 이미지 파일:
  
  ```{Dockerfile}
  # https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist 해당 링크에서 호스트 서버에 해당하는 cuda 버전과 ubuntu 버전을 확인할 수 있습니다.
  # 버전이 확인되었다면 cuda:00.0.0-cudnn-devel-ubuntu00.00으로 대신 입력하면 됩니다. 가능하면 호스트 서버와 동일한 cuda/ubuntu 버전으로 세팅해주세요.
  FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
  RUN apt-get update
  RUN apt-get install -y openssh-server
  RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
  RUN ssh-keygen -A
  RUN mkdir -p /run/sshd
  RUN echo 'root:0000' | chpasswd
  
  CMD ["/usr/sbin/sshd", "-D"]
  ```

  * 이미지 빌드 및 도커 실행: 현재 디렉토리에 위의 이미지 파일이 `Dockerfile` 이름으로 저장되었을 때 순차적으로 실행

  ```{command}
  sudo docker build -t test-image -f Dockerfile .
  ```
  
  ```
  sudo docker run -itd \
	--name test-container \
	-p 8888:22 \
	--gpus all \
	--restart=unless-stopped \
	--shm-size=32g \
	--ipc=host \
	test-image
  ```

  > 빌드 시 `-t`에 `test-image`를 임의의 이름으로 바꿔서 설정하시면 되며, 도커 실행 시 `--name` 또한 임의로 지정할 수 있습니다.
  > 
  > `-p`: `컨테이너 포트(임의로 큰 수를 설정):호스트 포트(보통 22)`
  > 
  > `--restart`: PC 재시작 시 도커 자동 실행
  >
  > `--shm-size`: 공유 메모리 설정 (기본값으로 설정했을 때, 너무 작아 오류가 생길 수 있으므로 32g로 설정함)
  > 
  > `--ipc`: 공유 메모리 네임스페이스 (호스트에서 가져오기)


## 4. Data Processing and Training

&nbsp;해당 리포지토리에는 총 세 개의 쉘 스크립트가 있습니다.


## Extra) [FSDP-QLoRA] Multi-GPU with QLoRA

* [Fully Sharded Data Parallel](https://huggingface.co/docs/peft/main/en/accelerate/fsdp#use-peft-qlora-and-fsdp-for-finetuning-large-models-on-multiple-gpus)
* [FSDP-QLoRA](https://huggingface.co/docs/bitsandbytes/main/fsdp_qlora)
* Multi-GPU 환경에서는 accelerator 모듈을 이용하여 분산 학습을 수행해야 합니다.
* 양자화 없이 학습하는 경우 FSDP를 아무런 문제 없이 바로 작동시킬 수 있습니다. 하지만 QLoRA의 경우 특정한 방법론을 적용하기 위해 코드를 약간 수정해야 합니다. 위 두 개 링크를 참고해주세요.
* 분산 환경에서 `SaveInferenceResultsCallback`이 정상적으로 동작하지 않음을 확인했습니다. 분산 학습 시 제외합니다.

## 기타 메모

* 훈련 데이터셋 규모가 너무 작음. 일반화에 어려움을 겪을 가능성 높음
* 적어도 수치값에 한정해서는 아예 없는 숫자를 지어내서 제시하지는 않는듯

## QLoRA Alignment 학습 시 유의사항

* 해당 링크에서 발췌: https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997
* 병합 이후 다시 양자화하여 DPO adpater를 부착, 학습할 경우 SFT 모델이 왜곡됨. 따라서 모델을 병합하지 않고, SFT adapter를 DPO로 튜닝하는 것이 가장 효과적인 방법임.
* 최종 DPO를 마친 모델은 병합하여 추론하는 것이 바람직하나, 여러 시도를 해봤음에도 결과가 왜곡되었음. 직접 구현하면 방법이 있을지도... 참고 문서가 거의 없음.
* QLoRA로 학습 후 어뎁터와 병합할 경우, 결과가 뭉개짐. 아직 효과적인 QLoRA merge를 지원하는 공식적인 방법은 없?음.
* 추론 및 빌드는 어뎁터를 병합하지 않았더라도 vllm을 무조건 활용하세요. 처음 컴파일 하는데 시간이 많이 걸리긴 하지만, cpp 기반 + 유동 배치 활용이라는 점에서 몇백배는 더 빠른 퍼포먼스를 보여줍니다.
