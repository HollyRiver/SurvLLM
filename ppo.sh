## GPU가 여럿 장착되어 있을 경우에 CUDA_VISIBLE_DEVICE를 앞에 붙여줄 것

## 선호도 데이터셋이 준비되었을 때 사용
nohup python RM.py --config config/RM_config_v1.1.2.8A.yaml > logs/rm_log_v1.1.2.8A.txt &

## ppo 데이터셋 준비
nohup python utils/csv_to_json_dataset.py --target="data/ppo_dataset.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt"\
                                    --ppo=true &

## 단순 입력 프롬프트만 존재하는 데이터 사용
nohup python PPO.py --config config/PPO_config_v1.1.2.12H.yaml > logs/ppo_log_v1.1.2.12H.txt &

## PPO에서 온전한 모델을 픽스하고, 양자화된 base model이 따로 저장되었으며, 추론에 사용할 프롬프트가 준비되었을 때
nohup python vllm_inference.py --base_model_path="base_model/Llama-3.1-8B-Instruct-nf4"\
                               --adapter_path="adapter/PPO-Llama-v1.1.2.15Am"\
                               --inference_data="data/inference_data.json"\
                               --output_dir="inference/inference_PPO_RLAIF_minor_v1.1.2.15.csv"\
                               --sampling=True\
                               --repetition_penalty=1.0\
                               --gpu_memory_util=0.9\
                               --seed=42 &

## (선택) 어댑터 병합: 추론 속도용. GPU 불필요, CPU/RAM으로 1분 내외.
## nf4 베이스를 fp32로 역양자화하고, LoRA delta를 fp32로 더한 뒤 fp16으로 저장합니다. 완전 무손실이 필요하면 --dtype fp32를 사용하세요.
## 어댑터 경로 규약은 vllm_inference.py와 동일 ('sft' 포함 시 그대로, 아니면 policy/ 하위). 출력 폴더는 비어 있어야 합니다.
nohup python utils/merge_qlora_adapter.py --nf4_base="base_model/Llama-3.1-8B-Instruct-nf4"\
                                          --adapter="adapter/PPO-Llama-v1.1.2.15H"\
                                          --output="merged_model/PPO-Llama-v1.1.2.15H-fp16"\
                                          --dtype="fp16" > logs/merge_ppo_v1.1.2.15H.txt &

## 병합된 모델 추론(기본값)
## 미병합 시 LoRA 저장 위치를 --adapter_path에 추가
nohup python vllm_inference.py --base_model_path="merged_model/PPO-Llama-v1.1.2.15H-fp16"\
                               --adapter_path=""\
                               --dtype="float16"\
                               --inference_data="data/inference_data.json"\
                               --output_dir="inference/inference_PPO_RLAIF_v1.1.2.15H_merged.csv"\
                               --sampling=True\
                               --repetition_penalty=1.0\
                               --gpu_memory_util=0.9\
                               --seed=42 &

## 병합 결과 검증 (CPU): 미병합 nf4+LoRA 대비 KL/greedy 비교. 역양자화 베이스가 필요하므로 먼저 --dequant_only로 생성
# CUDA_VISIBLE_DEVICES="" python utils/merge_qlora_adapter.py --dequant_only --output="merged_model/Llama-3.1-8B-Instruct-nf4-dequant-bf16" --dtype="bf16"
# CUDA_VISIBLE_DEVICES="" python debug/compare_merged_vs_lora.py --variants ref_bf16 merged_fp16 --samples 8 7 9