## exclude는 리스트 값을 넣을 수 있음: --exclude Plt Hct
# python dropna.py --inference SFT_v1.1.2.csv\
#                  --discharge data_all_20260205.csv\
#                  --save-path ../inference_dataset\
#                  --exclude Plt &

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

CLINICAL_VARS = [
    "sbp", "dbp",
    "hr", "rr", "temp",
    "wbc", "rbc", "hgb", "hct", "plt",
    "mcv", "mch", "mchc", "rdw", "glucose",
]


# ── Parsing functions ──────────────────────────────────────────────────

def extract_sex(text: str) -> float:
    if not isinstance(text, str):
        return float("nan")
    match = re.search(r"\b(male|female)\b", text.lower())
    if not match:
        return float("nan")
    return 1.0 if match.group(1) == "male" else 0.0


def extract_value(text: str, label: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    pattern = rf"(?i){re.escape(label)}\s*([0-9]+(?:\.[0-9]+)?(?:[-/][0-9]+(?:\.[0-9]+)?)*)"
    match = re.search(pattern, text)
    if not match:
        return None
    return re.split(r"[-/]", match.group(1))[-1]


def extract_bp(text: str) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(text, str):
        return None, None
    match = re.search(r"BP\s*([0-9A-Za-z./-]+)", text.upper())
    if not match:
        return None, None
    clean = re.sub(r"[^0-9./-]", "", match.group(1))
    parts = clean.split("/")
    if len(parts) != 2:
        return None, None
    sbp = parts[0].split("-")[-1] or None
    dbp = parts[1].split("-")[-1] or None
    return sbp, dbp


def parse_row(text: str) -> dict:
    sbp_raw, dbp_raw = extract_bp(text)
    return {
        "sex":         extract_sex(text),
        "sbp_raw":     sbp_raw,
        "dbp_raw":     dbp_raw,
        "hr_raw":      extract_value(text, "HR"),
        "rr_raw":      extract_value(text, "RR"),
        "temp_raw":    extract_value(text, "TEMP"),
        "wbc_raw":     extract_value(text, "WBC"),
        "rbc_raw":     extract_value(text, "RBC"),
        "hgb_raw":     extract_value(text, "HGB"),
        "hct_raw":     extract_value(text, "HCT"),
        "plt_raw":     extract_value(text, "PLT"),
        "mcv_raw":     extract_value(text, "MCV"),
        "mch_raw":     extract_value(text, "MCH"),
        "mchc_raw":    extract_value(text, "MCHC"),
        "rdw_raw":     extract_value(text, "RDW"),
        "glucose_raw": extract_value(text, "GLUCOSE"),
    }


# ── Report helpers ─────────────────────────────────────────────────────

def normalize_exclude(exclude: Optional[Iterable[str]]) -> set[str]:
    excluded = {v.strip().lower() for v in (exclude or []) if v and v.strip()}
    unknown = excluded - set(CLINICAL_VARS)
    if unknown:
        raise ValueError(
            f"Unknown covariates in --exclude: {sorted(unknown)}. Allowed: {CLINICAL_VARS}"
        )
    return excluded


def compute_nafl(df: pd.DataFrame, excluded: set[str]) -> pd.Series:
    check_cols = [c for c in df.columns if c not in excluded]
    return df[check_cols].isna().any(axis=1).map({True: "Y", False: "N"})


def build_out_name(inference_name: str, excluded: set[str]) -> str:
    p = Path(inference_name)
    if not excluded:
        return f"dataset_n_{p.name}"
    return f"dataset_n_{p.stem}_excl_{'_'.join(sorted(excluded))}{p.suffix}"


def print_na_summary(sampling: pd.DataFrame) -> None:
    na_cols = ["sex"] + CLINICAL_VARS
    summary = pd.DataFrame({
        "variable": na_cols,
        "na_count": sampling[na_cols].isna().sum().values,
        "na_pct":   (sampling[na_cols].isna().mean() * 100).round(2).values,
    })
    print("\nNA summary:")
    print(summary.to_string(index=False))


def print_death_event_dist(df: pd.DataFrame, label: str) -> None:
    counts = df.groupby("death_event", dropna=False).size().reset_index(name="n")
    counts["pct"] = (counts["n"] / counts["n"].sum() * 100).round(2)
    print(f"\nDeath event distribution ({label}):")
    print(counts.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse LLM-generated clinical text and produce a complete-case dataset."
    )
    parser.add_argument(
        "-i", "--inference",
        required=True,
        metavar="FILENAME",
        help="inference CSV filename (e.g. SFT_v1.1.2.csv); looked up as <save-path>/inference_<FILENAME>",
    )
    parser.add_argument(
        "-d", "--discharge",
        required=True,
        metavar="PATH",
        help="path to the discharge dataset CSV (e.g. data_all_20260205_수정01.csv)",
    )
    parser.add_argument(
        "-s", "--save-path",
        default="../inference_dataset",
        metavar="DIR",
        help="directory containing inference CSVs and where output is written (default: ../inference_dataset)",
    )
    parser.add_argument(
        "-e", "--exclude",
        nargs="*",
        default=None,
        metavar="VAR",
        help=(
            "covariates to exclude from NA filtering (case-insensitive, e.g. --exclude Plt Hct); "
            "rows with NaN in these columns are still kept as complete cases (default: None)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    save_path = Path(args.save_path)
    inference_name = args.inference

    dt = pd.read_csv(args.discharge, encoding="utf-8")
    a  = pd.read_csv(save_path / f"inference_{inference_name}", encoding="utf-8")

    # Parse LLM outputs
    parsed  = a["generated_text"].apply(parse_row).apply(pd.Series)
    dataset = pd.concat([a[["subject_id", "generated_text"]], parsed], axis=1)

    for var in CLINICAL_VARS:
        dataset[var] = pd.to_numeric(dataset[f"{var}_raw"], errors="coerce")

    # Join with discharge data
    sampling = (
        dataset[["subject_id", "sex"] + CLINICAL_VARS + ["generated_text"]]
        .merge(
            dt[["subject_id", "survival_time", "death_event", "text"]],
            on="subject_id",
            how="left",
        )
        [["subject_id", "sex"] + CLINICAL_VARS + ["survival_time", "death_event", "generated_text", "text"]]
    )

    # 0 → NaN (clinically impossible values)
    sampling[CLINICAL_VARS] = sampling[CLINICAL_VARS].replace(0, float("nan"))

    # Row-level completeness flag (skips covariates listed in --exclude)
    excluded = normalize_exclude(args.exclude)
    sampling["nafl"] = compute_nafl(sampling, excluded)

    # Save complete cases
    sampling_n = sampling[sampling["nafl"] == "N"].copy()
    out_path = save_path / build_out_name(inference_name, excluded)
    sampling_n.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Complete cases (nafl == N): {len(sampling_n)}")
    print(f"Saved → {out_path}")

    # Reports
    print_na_summary(sampling)
    print_death_event_dist(sampling_n, "complete cases")
    print_death_event_dist(sampling,   "all cases")

    print("\nSurvival time summary (complete cases):")
    print(sampling_n["survival_time"].describe().round(2))


if __name__ == "__main__":
    main()
