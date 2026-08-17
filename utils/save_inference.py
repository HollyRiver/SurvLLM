import os
import torch
import pandas as pd

from transformers import TrainerCallback
from trl import SFTTrainer


class SaveInferenceResultsCallback(TrainerCallback):
    def __init__(self, trainer: SFTTrainer, test_dataset, model_name):
        super().__init__()
        self.trainer = trainer 
        self.test_dataset = test_dataset
        self.output_dir = f"logs/{model_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        ## Multi-GPU 사용 시 메인 프로세스에서만 실행되도록 하여 중복 저장을 방지
        if state.is_world_process_zero:
            epoch = int(state.epoch) # 현재 epoch 번호
            output_path = os.path.join(self.output_dir, f"epoch_{epoch}_results.csv")
            
            print(f"\nEpoch {epoch} 종료. 테스트 데이터셋 추론 시작...")
            
            ## 현재 모델 획득, 추론 모드로 설정
            model = self.trainer.model.eval()
            tokenizer = self.trainer.tokenizer

            results = []

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for idx in range(self.test_dataset.num_rows):
                        messages = self.test_dataset[idx]["messages"][:2]
                        subject_id = self.test_dataset[idx]["subject_id"]

                        input_ids = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt"
                        ).to(model.device)

                        terminators = [
                            tokenizer.eos_token_id,
                        ]

                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=512,
                            eos_token_id=terminators,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=False ## greedy search
                        )
                        
                        response = outputs[0][input_ids.shape[-1]:]
                        generation = tokenizer.decode(response, skip_special_tokens=True)
                        results.append({"subject_id": subject_id, "generation": generation})

            # epoch별 파일 저장
            pd.DataFrame(results).to_csv(output_path, index = False, encoding = "utf-8-sig")
            
            print(f"Epoch {epoch} 추론 결과 저장 완료: {output_path}")