import pandas as pd

from check_covariate_format import (
    COVARIATES, STATUS_BAD_VALUE, STATUS_MISSING, STATUS_OK, check_frame, check_text,
)

# 합성 예시 (실제 데이터 아님)
GOOD = (
    "A female patient presented with pain; on physical examination, BP 120/80, HR 90, "
    "RR NA, and Temp 98.1; admission labs showed WBC 5.0, RBC 4.2, Hgb 14.0, Hct 42.0, "
    "Plt 200, MCV 90, MCH 30.0, MCHC 33.0, RDW 14.0, and Glucose 110."
)


def test_well_formed_text_is_all_ok():
    assert set(check_text(GOOD).values()) == {STATUS_OK}


def test_bp_na_is_ok_but_single_number_is_bad():
    assert check_text(GOOD.replace("BP 120/80", "BP NA"))["BP"] == STATUS_OK
    assert check_text(GOOD.replace("BP 120/80", "BP 120"))["BP"] == STATUS_BAD_VALUE


def test_mch_missing_is_not_satisfied_by_mchc():
    result = check_text(GOOD.replace("MCH 30.0, ", ""))
    assert result["MCH"] == STATUS_MISSING
    assert result["MCHC"] == STATUS_OK


def test_value_glued_to_unit_or_text_is_bad():
    assert check_text(GOOD.replace("Temp 98.1", "Temp 98.1F"))["Temp"] == STATUS_BAD_VALUE
    assert check_text(GOOD.replace("HR 90", "HR elevated"))["HR"] == STATUS_BAD_VALUE


def test_sex_requires_male_or_female_word():
    assert check_text(GOOD.replace("female patient", "patient"))["Sex"] == STATUS_MISSING
    assert check_text(GOOD.replace("female", "Male"))["Sex"] == STATUS_OK


def test_covariate_names_are_case_insensitive():
    result = check_text(GOOD.replace("Hct 42.0", "HCT 42.0").replace("Glucose", "glucose"))
    assert result["Hct"] == STATUS_OK
    assert result["Glucose"] == STATUS_OK


def test_non_string_is_all_missing():
    assert set(check_text(float("nan")).values()) == {STATUS_MISSING}


def test_check_frame_shape():
    status = check_frame(pd.DataFrame({"generated_text": [GOOD, None]}))
    assert status.shape == (2, len(COVARIATES))
