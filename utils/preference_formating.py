# 프로젝트 루트에서 실행:
# python utils/preference_formating.py data/Preference_AIF_Llama_v1.1.2_hf200_labels.csv \
#     --text_source=data/Preference_HF_Llama_v1.1.2.csv \
#     --output=data/Preference_AIF_Llama_v1.1.2m.csv

"""AIF 순위 라벨을 pairwise 선호도 데이터로 변환하는 코드

HF 라벨(5자리 순열 Label)용 원본 preference_formating.py를 AIF 라벨 형식에 맞춰 수정한
버전입니다. 원본은 다른 PC에 보존되어 있습니다.

입력 CSV(extract_aif_labels.py 산출물)는 subject_id 당 한 행으로 response_1 ~ response_5와
labeling_json({"Label 1": 순위, ..., "Label 5": 순위}, 1이 최선)을 갖습니다.

각 행에서 최대 C(5, 2) = 10개의 (chosen, rejected) 쌍을 만들되:
    - 순위가 동률(tie)인 쌍은 선호 방향을 정할 수 없으므로 제외
    - 텍스트가 완전히 동일한 쌍(chosen == rejected)은 선호 신호가 없으므로 제외
    - 동일 생성문 때문에 중복 발생한 (chosen, rejected) 쌍은 한 개만 유지 (원본과 동일)

--text_source(예: HF 선호도 csv)에서 subject_id별 text(퇴원요약지)를 결합해
Preference_HF_Llama와 동일한 열 구성으로 저장합니다:
    subject_id, text, chosen, chosen_ranking, rejected, rejected_ranking

chosen_ranking / rejected_ranking은 HF 파일과 동일하게 0-indexed 순위(0이 최선)입니다.
생성문에 섞인 CRLF는 HF 라벨링 파일과 동일하게 LF로 정규화합니다.
"""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from extract_aif_labels import GEN_COLS, parse_rank_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "AIF 순위 라벨(labeling_json)을 pairwise 선호도 데이터로 변환합니다."
    )
    parser.add_argument(
        "input",
        type = Path,
        help = "라벨 CSV 경로 (열: subject_id, response_1~5, labeling_json)",
    )
    parser.add_argument(
        "--text_source",
        type = Path,
        required = True,
        help = "subject_id별 text(퇴원요약지)를 가져올 CSV (예: HF 선호도 csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type = Path,
        default = None,
        help = "출력 CSV 경로. 기본값은 '<input stem> Preference.csv'",
    )
    return parser.parse_args()


def normalize_newlines(text: str) -> str:
    """생성문에 드물게 섞인 CRLF/CR을 LF로 통일해 HF 라벨링 파일의 텍스트와 일치시킨다."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_text_by_subject(path: Path) -> dict[int, str]:
    """text_source CSV에서 subject_id -> text 매핑을 만든다."""
    df_text = pd.read_csv(path, usecols = ["subject_id", "text"])

    ## 같은 subject_id에 서로 다른 text가 있으면 keep='first'가 조용히 나머지를 버리므로 즉시 실패
    conflicts = df_text.groupby("subject_id")["text"].nunique()
    conflict_ids = conflicts.index[conflicts > 1].tolist()
    if conflict_ids:
        raise ValueError(f"text_source에 subject_id당 text가 여러 개인 subject가 있습니다: {conflict_ids[:5]}")

    df_unique = df_text.drop_duplicates("subject_id")
    return dict(zip(df_unique["subject_id"], df_unique["text"]))


def row_to_ranks(labeling_json: str, subject_id: object) -> dict[int, int]:
    """labeling_json을 생성문 번호(1~5) -> 순위 dict로 변환. 유효하지 않으면 즉시 실패."""
    if pd.isna(labeling_json):
        raise ValueError(f"subject_id {subject_id}: labeling_json이 비어 있습니다 (NaN).")

    label = parse_rank_label(labeling_json)

    if label is None:
        raise ValueError(
            f"subject_id {subject_id}: 유효하지 않은 labeling_json입니다: {labeling_json!r}"
        )

    return {i: label[f"Label {i}"] for i in GEN_COLS}


def build_preference_pairs(
    df_labels: pd.DataFrame, text_by_subject: dict[int, str]
) -> tuple[pd.DataFrame, int, int]:
    """subject별 순위 라벨에서 (chosen, rejected) 쌍을 생성한다.

    반환: (선호도 쌍 DataFrame, tie로 제외된 쌍 수, 동일 텍스트로 제외된 쌍 수)
    """
    rows: list[dict] = []
    n_tie_skipped = 0
    n_identical_skipped = 0

    for _, r in df_labels.iterrows():
        subject_id = r["subject_id"]

        if subject_id not in text_by_subject:
            raise ValueError(f"subject_id {subject_id}: text_source에 text가 없습니다.")

        ranks = row_to_ranks(r["labeling_json"], subject_id)

        missing_cols = [i for i in GEN_COLS if pd.isna(r[f"response_{i}"])]
        if missing_cols:
            raise ValueError(f"subject_id {subject_id}: response_{missing_cols} 열이 비어 있습니다 (NaN).")

        responses = {i: normalize_newlines(r[f"response_{i}"]) for i in GEN_COLS}

        for gen_a, gen_b in itertools.combinations(GEN_COLS, 2):
            ## 동률이면 어느 쪽이 chosen인지 정할 수 없으므로 쌍 자체를 만들지 않는다
            if ranks[gen_a] == ranks[gen_b]:
                n_tie_skipped += 1
                continue

            chosen_gen, rejected_gen = (
                (gen_a, gen_b) if ranks[gen_a] < ranks[gen_b] else (gen_b, gen_a)
            )

            ## 완전히 동일한 생성문 쌍은 선호 신호가 없으므로 제외 (원본과 동일)
            if responses[chosen_gen] == responses[rejected_gen]:
                n_identical_skipped += 1
                continue

            rows.append(
                {
                    "subject_id": subject_id,
                    "text": text_by_subject[subject_id],
                    "chosen": responses[chosen_gen],
                    "chosen_ranking": ranks[chosen_gen] - 1,
                    "rejected": responses[rejected_gen],
                    "rejected_ranking": ranks[rejected_gen] - 1,
                }
            )

    return pd.DataFrame(rows), n_tie_skipped, n_identical_skipped


def main() -> None:
    args = parse_args()

    df_labels = pd.read_csv(args.input, encoding = "utf-8-sig")

    ## subject_id가 중복되면 같은 쌍이 두 번 생성된 뒤 dedup에 조용히 흡수되므로 입력 단계에서 즉시 실패
    if not df_labels["subject_id"].is_unique:
        dup_ids = df_labels.loc[df_labels["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"입력 CSV에 중복된 subject_id가 있습니다: {dup_ids[:5]}")

    text_by_subject = load_text_by_subject(args.text_source)

    tidy, n_tie, n_identical = build_preference_pairs(df_labels, text_by_subject)

    ## 동일 생성문 탓에 여러 행으로 나타난 같은 (chosen, rejected) 쌍은 한 개만 유지 (원본과 동일)
    before = len(tidy)
    tidy = tidy.drop_duplicates(subset = ["chosen", "rejected"]).reset_index(drop = True)
    n_duplicate = before - len(tidy)

    output = args.output or args.input.with_name(f"{args.input.stem} Preference.csv")
    tidy.to_csv(output, index = False)
    print(
        f"저장 완료: {output} ({len(tidy)}쌍, subject {tidy['subject_id'].nunique()}건) | "
        f"제외 내역 - tie: {n_tie}쌍, 동일 텍스트: {n_identical}쌍, 중복: {n_duplicate}쌍"
    )


if __name__ == "__main__":
    main()
