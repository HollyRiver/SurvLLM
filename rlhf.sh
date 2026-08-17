## 여기 있는거 한번에 다 안돼요. 성능상 한번에 해도 안되고.
## 그리고 실행할 때에는 &&가 아닌 &를 붙여야 합니다. 여긴 그냥 순차적으로 된다는 가정하에 작성했어요.

nohup python utils/csv_to_json_dataset.py --target="data/data_sample1_for_SFT_20260205.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &&

nohup python SFT.py --config config/SFT_config_v1.2.0.yaml > logs/sft_log_v1.2.0.txt &&
## multi GPU 사용 시 fsdp_config_qlora.yaml 파일에서 num_processes에 GPU 숫자만 수정하여 아래를 전부 입력
# nohup env \
# accelerate launch --config_file "config/fsdp_config_qlora.yaml" \
# SFT.py --config config/SFT_config_multi_GPU.yaml > sft_test.log &

## generated_data for DPO를 생성
nohup python utils/csv_to_json_dataset.py --target="data/dpo_prompt_data.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &

## 양자화 모델 생성
nohup python utils/gen_llama_nf4.py &&

## SFT에서 온전한 모델을 픽스하고, 양자화된 base model이 따로 저장되었으며, 추론에 사용할 프롬프트가 준비되었을 때
## !!!어댑터 저장 폴더에는 무조건 "sft"라는 키워드를 넣어주세요, 그래야 인식합니다!!!
nohup python vllm_inference.py --base_model_path="base_model/Llama-3.1-8B-Instruct-nf4"\
                               --adapter_path="adapter/Zip-Llama-sft-v1.2.0"\
                               --inference_data="data/dpo_prompt_data.json"\
                               --output_dir="data/generated_data_v1.2.0.csv"\
                               --gen_nums=5\
                               --sampling=True\
                               --temperature=1.0\
                               --repetition_penalty=1.0\
                               --gpu_memory_util=0.9\
                               --seed=42 &

## SFT 완료 모델 추론 결과를 LLM으로 선호도 레이블링
nohup python preference_AIF.py --model_name="Qwen/Qwen3-30B-A3B"\
                               --preference_name="data/generated_data_v1.2.0.csv"\
                               --discharge_name="data/dpo_prompt_data.csv" &

## SFT에서 온전한 모델을 픽스하고, 데이터셋이 준비되었을 때
nohup python utils/csv_to_json_dataset.py --target="data/Preference_AIF_Llama_v1.1.2m.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt"\
                                    --name_tag="_AIF_minor" &&

nohup python DPO.py --config config/DPO_config_v1.1.2.2A.yaml > logs/dpo_log_v1.1.2.2A.txt &&
## Multi-GPU를 사용할 경우: Adapter Twice Load로 일단 되긴 하는데, 메모리 더 많이 먹음...
# nohup env \
# NCCL_TIMEOUT=600 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --config_file "config/fsdp_config_qlora_dpo.yaml" \
# DPO.py --config config/DPO_config_multi_GPU.yaml > dpo_test.log &

nohup python utils/csv_to_json_dataset.py --target="data/inference_data.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &&

## DPO에서 온전한 모델을 픽스하고, 양자화된 base model이 따로 저장되었으며, 추론에 사용할 프롬프트가 준비되었을 때
nohup python vllm_inference.py --base_model_path="base_model/Llama-3.1-8B-Instruct-nf4"\
                               --adapter_path="adapter/DPO-Llama-v1.1.2.2Am"\
                               --inference_data="data/inference_data.json"\
                               --output_dir="inference/inference_DPO_RLAIF_minor_v1.1.2.2.csv"\
                               --sampling=True\
                               --repetition_penalty=1.0\
                               --gpu_memory_util=0.9\
                               --seed=42 &