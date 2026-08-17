# 생존 분석을 위한 텍스트 언어 모델 튜닝

* [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 모델을 튜닝하여 장문의 텍스트에서 중요한 단서를 1차적으로 추출
* [QLoRA](https://arxiv.org/abs/2305.14314), [Load to adapter twice](https://huggingface.co/docs/trl/dpo_trainer#using-option-3---load-the-adapter-twice) 사용


## Setup
* 100GB의 VRAM 및 200GB의 CPU RAM (권장). 입력되는 최대 시퀀스 길이를 더 길게 설정한다면 이보다 많이 필요합니다. (해당 문서에서는 16,384)
* 구축된 아나콘다 환경 (cuda 12.8, Ubuntu 20.04에서 구동시켰으나, Ubuntu 22.04 이상을 권장합니다. Ubuntu 20.04 버전에서는 flash-attention 실행을 위한 다운그레이드 및 GLibc 업데이트가 필요합니다.)
* Dependencies installation: `pip install transformers bitsandbytes datasets sentencepiece accelerate trl peft wandb openai pqdm`, pytorch와 flash-attention, vllm 설치는 부가적으로 수행해주세요. flash-attention-2가 사용되었습니다.

```
conda env create -f LLM.ymal
conda activate LLM
```

## FSDP-QLoRA

* [Fully Sharded Data Parallel](https://huggingface.co/docs/peft/main/en/accelerate/fsdp#use-peft-qlora-and-fsdp-for-finetuning-large-models-on-multiple-gpus)
* [FSDP-QLoRA](https://huggingface.co/docs/bitsandbytes/main/fsdp_qlora)
* Multi-GPU 환경에서는 accelerator 모듈을 이용하여 분산 학습을 수행해야 합니다.
* 양자화 없이 학습하는 경우 FSDP를 아무런 문제 없이 바로 작동시킬 수 있습니다. 하지만 QLoRA의 경우 특정한 방법론을 적용하기 위해 코드를 약간 수정해야 합니다. 위 두 개 링크를 참고해주세요.
* 분산 환경에서 `SaveInferenceResultsCallback`이 정상적으로 동작하지 않음을 확인했습니다. 분산 학습 시 제외합니다.

## 기타

* 시스템 프롬프트에 어떤 값을 가져와야 하는지를 조금 더 명시해야 한다고 판단됨
* 훈련 데이터셋 규모가 너무 작음. 일반화에 어려움을 겪을 가능성 높음
* text 데이터에 체온 화씨/섭씨 혼용되고 있음 -> 섭씨 온도 출력: 34.0°C
* 적어도 수치값에 한정해서는 없는 값을 지어내서 제시하지는 않는듯
* Glucose를 많이 틀림

## DPO 실험

### 기타 (https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997)

* 병합 이후 다시 양자화하여 DPO adpater를 부착, 학습할 경우 SFT 모델이 왜곡됨. 따라서 모델을 병합하지 않고, SFT adapter를 DPO로 튜닝하는 것이 가장 효과적인 방법임.
* 최종 DPO를 마친 모델은 병합하여 추론하는 것이 바람직하나, 여러 시도를 해봤음에도 결과가 왜곡되었음. 스크래치로 구현하면 방법이 있을지도... 참고 문서가 거의 없음.
* QLoRA로 학습 후 어뎁터와 병합할 경우, 결과가 뭉개짐. 아직 효과적인 QLoRA merge를 지원하는 공식적인 방법은 없?음.
* 추론 및 빌드는 어뎁터를 병합하지 않았더라도 vllm을 무조건 활용하세요. 처음 컴파일 하는데 시간이 많이 걸리긴 하지만, cpp 기반 + 유동 배치 활용이라는 점에서 몇백배는 더 빠른 퍼포먼스를 보여줍니다.