# python extract_aif_labels.py --raw_name="data/Preference_AIF_Llama_v1.1.2_raw.csv" \
#                               --generated_name="data/generated_data_v1.1.2.csv" \
#                               --reference_name="data/Preference_HF_Llama_v1.1.2.csv" \
#                               --output_name="data/Preference_AIF_Llama_v1.1.2_hf200_labels.csv"

"""
AIF 판정 모델의 raw 출력에서 subject_id별 순위 JSON을 추출해 분석용 CSV를 만드는 코드

    1. raw csv(subject_id, attempt, finish_reason, raw_output)에서 순위 JSON을 추출
       - 재시도(attempt) 회차가 여러 개인 subject는 유효 JSON이 추출되는 마지막 회차를 채택
    2. long format 생성 데이터에서 Label 1~5 인덱스에 대응하는 응답 5개를 wide format으로 결합
    3. --reference_name이 주어지면 해당 파일의 subject_id 집합으로 필터링 (예: HF 라벨링 200건과 비교)

출력 csv 열: subject_id, response_1 ~ response_5, labeling_json

순위 JSON 추출/검증 로직은 preference_AIF.py와 동일합니다. preference_AIF.py는 최상단에서
vllm을 import하므로 (vLLM 미설치 환경에서도 돌도록) import 대신 로직을 복제해 두었습니다.
preference_AIF.py의 판정 규칙을 수정하면 이 파일도 함께 맞춰야 합니다.
"""
import pandas as pd

import json
import re
import os
import argparse

N_GENERATIONS = 5                       ## subject_id 당 생성문 개수
GEN_COLS = list(range(1, N_GENERATIONS + 1))
LABEL_KEYS = [f"Label {i}" for i in GEN_COLS]   ## 판정 JSON의 키


def parse_rank_label(json_text: str) -> dict[str, int] | None:
    """JSON 문자열을 순위 dict로 파싱. 유효하지 않으면 None 반환. (preference_AIF.py와 동일)"""
    try:
        label = json.loads(json_text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(label, dict) or set(label.keys()) != set(LABEL_KEYS):
        return None

    values = list(label.values())

    ## JSON true/false는 파이썬에서 int의 하위 타입(bool)이므로 명시적으로 거른다
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return None
    ## 허용 범위(1~5)를 벗어난 순위는 제외
    if min(values) < 1 or max(values) > N_GENERATIONS:
        return None
    ## 모든 순위가 동일하면 선호 비교가 불가능하므로 스킵. 부분 동률은 허용
    if len(set(values)) == 1:
        return None

    return label


def extract_rank_label(raw_text: str) -> dict[str, int] | None:
    """추론(reasoning) 토큰을 포함한 모델 출력에서 순위 JSON을 추출. (preference_AIF.py와 동일)"""
    ## 추론이 max_tokens에 잘려 </think>가 닫히지 않은 경우 무효 처리
    if "<think>" in raw_text and "</think>" not in raw_text:
        return None

    ## 추론 과정에 예시 JSON이 섞일 수 있으므로 </think> 이후 텍스트만 사용
    answer_text = raw_text.split("</think>")[-1]

    ## 중첩 없는 JSON 객체 후보를 모두 찾아 마지막 유효 후보를 채택
    candidates = re.findall(r"\{[^{}]*\}", answer_text)

    for candidate in reversed(candidates):
        label = parse_rank_label(candidate)

        if label is not None:
            return label

    return None


def build_wide_generations(df_gen: pd.DataFrame) -> pd.DataFrame:
    """long format 생성 데이터를 subject_id 당 생성문 5개를 열(1~5)로 갖는 wide format으로 변환. (preference_AIF.py와 동일)"""
    counts = df_gen.groupby("subject_id").size()
    invalid_ids = counts.index[counts != N_GENERATIONS]

    if len(invalid_ids) > 0:
        print(f"[Warning] 생성문이 {N_GENERATIONS}개가 아닌 subject_id {len(invalid_ids)}건 제외: {list(invalid_ids[:10])} ...")

    df_valid = df_gen.loc[df_gen["subject_id"].isin(counts.index[counts == N_GENERATIONS])]
    df_valid = df_valid.assign(gen_num = df_valid.groupby("subject_id").cumcount() + 1)

    return df_valid.pivot(index = "subject_id", columns = "gen_num", values = "generated_text").reset_index()


def resolve_data_path(name: str) -> str:
    """디렉토리 없이 파일 이름만 주어지면 data/ 폴더 기준으로 해석한다."""
    return name if os.path.dirname(name) else os.path.join("data", name)


def extract_labels(df_raw: pd.DataFrame, df_wide: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """raw 판정 출력에서 subject_id별 순위 JSON을 추출해 응답 5개 열과 결합한다."""
    records = []
    failed_ids = []

    for subject_id, group in df_raw.sort_values("attempt").groupby("subject_id"):
        ## 마지막 재시도 회차부터 역순으로 탐색해 첫 유효 JSON을 채택
        label = next(
            (lbl for raw in reversed(group["raw_output"].tolist()) if (lbl := extract_rank_label(raw))),
            None,
        )

        gen_row = df_wide.loc[df_wide["subject_id"] == subject_id]

        if label is None or gen_row.empty:
            failed_ids.append(subject_id)
            continue

        responses = {f"response_{i}": gen_row.iloc[0][i] for i in GEN_COLS}
        records.append({
            "subject_id": subject_id,
            **responses,
            "labeling_json": json.dumps(label, ensure_ascii = False),
        })

    return pd.DataFrame(records), failed_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_name", type = str, default = None, help = "AIF 판정 모델 raw 출력 csv (preference_AIF.py의 *_raw.csv)")
    parser.add_argument("--generated_name", type = str, default = None, help = "Label 인덱스의 원천인 long format 생성 데이터 csv")
    parser.add_argument("--reference_name", type = str, default = None, help = "subject_id 필터링 기준 csv (예: HF 라벨링 파일). 생략 시 전체 추출")
    parser.add_argument("--output_name", type = str, default = None, help = "출력 csv 경로")
    args = parser.parse_args()

    if args.raw_name is None or args.generated_name is None or args.output_name is None:
        parser.error("--raw_name, --generated_name, --output_name은 필수 인자입니다.")

    df_raw = pd.read_csv(resolve_data_path(args.raw_name))
    df_wide = build_wide_generations(pd.read_csv(resolve_data_path(args.generated_name)))

    if args.reference_name is not None:
        reference_ids = set(pd.read_csv(resolve_data_path(args.reference_name), usecols = ["subject_id"])["subject_id"].unique())
        df_raw = df_raw.loc[df_raw["subject_id"].isin(reference_ids)]
        print(f"기준 파일 subject_id {len(reference_ids)}건 중 raw에 존재: {df_raw['subject_id'].nunique()}건")

    df_out, failed_ids = extract_labels(df_raw, df_wide)

    if failed_ids:
        print(f"[Warning] 유효 순위 JSON 추출 실패 {len(failed_ids)}건: {failed_ids[:10]}")

    output_path = resolve_data_path(args.output_name)
    df_out.to_csv(output_path, index = False, encoding = "utf-8-sig")
    print(f"저장 완료: {output_path} ({df_out.shape[0]}행, {df_out.shape[1]}열)")
