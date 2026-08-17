## 선호도 데이터셋이 준비되었을 때 사용
nohup python RM.py --config config/RM_config_v1.1.2.8A.yaml > logs/rm_log_v1.1.2.8A.txt &

## ppo 데이터셋 준비
nohup python csv_to_json_dataset.py --target="data/ppo_dataset.csv"\
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