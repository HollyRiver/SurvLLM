"""추론 결과(generated_text)에 15개 공변량이 형식대로 들어있는지 검사한다.

검사 규칙 (퇴원요약지 원문 값과의 일치 여부는 보지 않음):
- 성별: 단어 단위로 Male 또는 Female 이 등장 (대소문자 무시, 문장 중간의 "female" 허용)
- BP: "BP <수치>/<수치>" 또는 "BP NA"
- 나머지 13개: "<이름> <수치>" 또는 "<이름> NA"
  (MCH 와 MCHC 는 단어 경계로 구분)
- 공변량 이름은 대소문자 무시 (HCT, hct 도 Hct 로 인정; utils/dropna.py 추출 규칙과 동일)

각 공변량 상태:
- ok        : 이름 + 올바른 값
- bad_value : 이름은 있으나 뒤따르는 값이 형식에 맞지 않음
- missing   : 이름 자체가 없음 (성별은 Male/Female 이 없으면 missing)

사용법 (프로젝트 루트에서):
    python analysis/check_covariate_format.py
결과:
    analysis/results/covariate_format_ok_rate_wide.csv — 공변량×모델 형식 준수율(%) wide 표
    analysis/results/covariate_format_summary.csv   — 파일×공변량 집계 (subject_id/원문 없음)
    analysis/results/covariate_format_failures.csv  — 실패 행 (subject_id 포함, git 제외)
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFERENCE_DIR = PROJECT_ROOT / "inference"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# 비교 기준 SFT + 가장 최신 DPO/PPO x RLHF/RLAIF/RLAIF_minor 결과
DEFAULT_TARGETS = (
    "inference_SFT_v1.1.2.csv",
    "inference_DPO_RLHF_v1.1.2.2.csv",
    "inference_DPO_RLAIF_v1.1.2.2.csv",
    "inference_DPO_RLAIF_minor_v1.1.2.2.csv",
    "inference_PPO_RLHF_v1.1.2.15.csv",
    "inference_PPO_RLAIF_v1.1.2.15.csv",
    "inference_PPO_RLAIF_minor_v1.1.2.15.csv",
)

TEXT_COLUMN = "generated_text"
ID_COLUMN = "subject_id"

SEX = "Sex"
NUMERIC_COVARIATES = (
    "HR", "RR", "Temp", "WBC", "RBC", "Hgb", "Hct", "Plt",
    "MCV", "MCH", "MCHC", "RDW", "Glucose",
)
COVARIATES = (SEX, "BP", *NUMERIC_COVARIATES)

STATUS_OK = "ok"
STATUS_BAD_VALUE = "bad_value"
STATUS_MISSING = "missing"

_NUMBER = r"-?\d+(?:\.\d+)?"
# 값 뒤에는 공백·구두점·문장 끝만 허용 ("98.1F", "107bpm" 같은 붙은 문자는 불합격)
_VALUE_END = r"(?=$|[\s,;:)\]]|\.(?:\s|$))"

SEX_PATTERN = re.compile(r"\b(?:male|female)\b", re.IGNORECASE)


# 공변량 이름은 대소문자 무시 (utils/dropna.py 의 extract_value 가 (?i) 로 추출하는 것과 일치)
def _name_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", re.IGNORECASE)


def _value_pattern(name: str, value_regex: str) -> re.Pattern:
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(name)}\s+(?:NA|{value_regex}){_VALUE_END}",
        re.IGNORECASE,
    )


NAME_PATTERNS = {name: _name_pattern(name) for name in COVARIATES if name != SEX}
VALUE_PATTERNS = {
    "BP": _value_pattern("BP", rf"{_NUMBER}\s*/\s*{_NUMBER}"),
    **{name: _value_pattern(name, _NUMBER) for name in NUMERIC_COVARIATES},
}


def check_text(text: object) -> dict[str, str]:
    """한 생성문에 대해 공변량별 상태 dict 를 반환한다."""
    if not isinstance(text, str):
        return {name: STATUS_MISSING for name in COVARIATES}

    sex_status = STATUS_OK if SEX_PATTERN.search(text) else STATUS_MISSING
    others = {
        name: (
            STATUS_OK if VALUE_PATTERNS[name].search(text)
            else STATUS_BAD_VALUE if NAME_PATTERNS[name].search(text)
            else STATUS_MISSING
        )
        for name in VALUE_PATTERNS
    }
    return {SEX: sex_status, **others}


def check_frame(df: pd.DataFrame) -> pd.DataFrame:
    """행별 공변량 상태 DataFrame (index 는 원본과 동일)."""
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"'{TEXT_COLUMN}' 컬럼이 없습니다: {df.columns.tolist()}")
    return pd.DataFrame(
        [check_text(t) for t in df[TEXT_COLUMN]], index=df.index, columns=list(COVARIATES)
    )


def summarize(file_name: str, status: pd.DataFrame) -> pd.DataFrame:
    """파일 하나에 대한 공변량별 ok/bad_value/missing 집계 (+ 15개 모두 ok 인 행 수)."""
    n_rows = len(status)
    rows = [
        {
            "file": file_name,
            "covariate": name,
            "n_rows": n_rows,
            STATUS_OK: int((status[name] == STATUS_OK).sum()),
            STATUS_BAD_VALUE: int((status[name] == STATUS_BAD_VALUE).sum()),
            STATUS_MISSING: int((status[name] == STATUS_MISSING).sum()),
        }
        for name in COVARIATES
    ]
    all_ok = int((status == STATUS_OK).all(axis=1).sum())
    rows.append({
        "file": file_name, "covariate": "ALL_15", "n_rows": n_rows,
        STATUS_OK: all_ok, STATUS_BAD_VALUE: None, STATUS_MISSING: None,
    })
    summary = pd.DataFrame(rows)
    return summary.assign(ok_rate=(summary[STATUS_OK] / n_rows).round(6))


def collect_failures(file_name: str, df: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """하나라도 ok 가 아닌 행을 (file, row, subject_id, issues, 생성문) 형태로 반환."""
    failed = status[~(status == STATUS_OK).all(axis=1)]
    issues = [
        "; ".join(f"{k}={v}" for k, v in row.items() if v != STATUS_OK)
        for _, row in failed.iterrows()
    ]
    ids = df.loc[failed.index, ID_COLUMN].to_numpy() if ID_COLUMN in df.columns else None
    return pd.DataFrame({
        "file": file_name,
        "row": failed.index,
        ID_COLUMN: ids,
        "issues": issues,
        TEXT_COLUMN: df.loc[failed.index, TEXT_COLUMN].to_numpy(),
    })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inference_dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_TARGETS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missing_files = [f for f in args.files if not (args.inference_dir / f).is_file()]
    if missing_files:
        print(f"[ERROR] 파일을 찾을 수 없습니다: {missing_files}", file=sys.stderr)
        return 1

    summaries, failures = [], []
    for file_name in args.files:
        df = pd.read_csv(args.inference_dir / file_name)
        status = check_frame(df)
        summaries.append(summarize(file_name, status))
        failures.append(collect_failures(file_name, df, status))
        print(f"[done] {file_name}: {len(df)} rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.output_dir / "covariate_format_summary.csv", index=False)
    pd.concat(failures, ignore_index=True).to_csv(
        args.output_dir / "covariate_format_failures.csv", index=False
    )

    wide = to_wide(summary, args.files)
    wide.to_csv(args.output_dir / "covariate_format_ok_rate_wide.csv")
    print(wide.to_string())
    return 0


def to_wide(summary: pd.DataFrame, file_order: list[str]) -> pd.DataFrame:
    """공변량(행) x 모델(열) 형식 준수율(%) 표. 열 이름은 'inference_'/'.csv' 를 뗀 모델명."""
    short = {f: f.removeprefix("inference_").removesuffix(".csv") for f in file_order}
    wide = (
        summary.pivot(index="covariate", columns="file", values="ok_rate")
        .reindex(index=[*COVARIATES, "ALL_15"], columns=file_order)
        .rename(columns=short)
        .mul(100)
        .round(1)
    )
    wide.columns.name = None
    return wide


if __name__ == "__main__":
    sys.exit(main())
