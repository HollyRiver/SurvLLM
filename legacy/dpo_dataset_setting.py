import pandas as pd
import datasets
from utils import remove_hangul


if __name__ == "__main__":
    ## 원시 데이터 로드
    df = pd.read_csv("data/gen_data_20251118_for_dpo.csv", encoding = "utf-8").\
        loc[lambda _df : _df.tie == "N"]
    df_text = pd.read_csv("data/data_sample_20251111_01.csv", encoding = "cp949")
    system_prompt = df_text.system[0]

    ds = datasets.Dataset.from_pandas(df)
    columns_to_remove = [f for f in list(ds.features) if f not in ["subject_id", "chosen", "rejected"]]

    ## Explicit format
    train_ds = ds.map(
        lambda sample: {
            "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sample["text"]}
                ],
            "chosen": [{"role": "assistant", "content": sample["chosen"]}],
            "rejected": [{"role": "assistant", "content": sample["rejected"]}]
        }
    )

    train_ds = train_ds.map(lambda sample: remove_hangul(sample, column = "prompt"))
    train_ds = train_ds.map(remove_columns = columns_to_remove, batched = False)
    train_ds = train_ds.train_test_split(test_size = 0.1, seed = 42)

    train_ds["train"].to_json("data/dpo_train_dataset.json", orient = "records")
    train_ds["test"].to_json("data/dpo_test_dataset.json", orient = "records")

    test_ds = train_ds["test"]

    lst = []

    for idx in range(test_ds.num_rows):
        lst.append({"subject_id": test_ds["subject_id"][idx], "chosen": test_ds[idx]["chosen"][0]["content"], "rejected": test_ds[idx]["rejected"][0]["content"], "text": test_ds[idx]["prompt"][1]["content"]})

    pd.DataFrame(lst).to_csv("data/dpo_test_label.csv", index = False, encoding = "utf-8-sig")