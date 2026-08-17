import pandas as pd
import datasets
from utils import remove_hangul


if __name__ == "__main__":
    ## 원시 데이터 로드
    df_text = pd.read_csv("data/data_sample_20251111_01.csv", encoding = "cp949")
    ds = datasets.Dataset.from_pandas(df_text)
    columns_to_remove = [f for f in list(ds.features) if f != "subject_id"]

    system_prompt = df_text.system[0]
    # system_prompt = "You are the world’s leading expert in survival analysis.\
    #  From a discharge summary, extract Chief Complaint, Physical Exam, and Admission Labs (Pertinent Results) and produce one sentence.\
    #  The sentence will be used for hazard calculation, so be precise, clinically accurate, and concise."
    # question = "Please summarize the following discharge summary\
    #  in one sentence focusing on Chief Complaint, Physical Exam, and Admission Labs."

    train_ds = ds.map(
        lambda sample:
        {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["text"]},
            {"role": "assistant", "content": sample["assistant"]}
        ]}
    )

    train_ds = train_ds.map(remove_columns = columns_to_remove, batched = False)
    train_ds = train_ds.map(lambda sample: remove_hangul(sample, column = "messages"))
    train_ds = train_ds.train_test_split(test_size = 0.1, seed = 42)

    train_ds["train"].to_json("data/sft_train_dataset.json", orient = "records")
    train_ds["test"].to_json("data/sft_test_dataset.json", orient = "records")

    test_ds = train_ds["test"]
    lst = []

    for idx in range(test_ds.num_rows):
        lst.append({"subject_id": test_ds["subject_id"][idx], "label": test_ds["messages"][idx][2]["content"], "text": test_ds["messages"][idx][1]["content"]})

    pd.DataFrame(lst).to_csv("data/test_label.csv", index = False, encoding = "utf-8-sig")