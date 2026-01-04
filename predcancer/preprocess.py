import math
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


from nan import count_nan, nan_imputation
from predcancer.settings import (CONTROL_RATIO, DATA_VERSION, DAYS_GAP,
                                 MATCH_DATE, PROJECT_DIR, TIME_SPLIT, USE_ED,
                                 USE_LAB, g_seed)

username = os.getlogin()

PREPROCESS_DIR = os.path.join(PROJECT_DIR, "preprocessed")
os.makedirs(PREPROCESS_DIR, exist_ok=True)

if MATCH_DATE:  # MATCH_DATE가 True인 경우: 1.pkl
    PREPROCESSED_FILE = os.path.join(PREPROCESS_DIR, f"{DATA_VERSION}_1.pkl")
else:  # MATCH_DATE가 False인 경우: 2.pkl
    PREPROCESSED_FILE = os.path.join(PREPROCESS_DIR, f"{DATA_VERSION}_2.pkl")

USE_OLDEST = DATA_VERSION.startswith("2025-10-09")
MIMIC_DIR = f"/home/{username}/data/mimic-iv-3.1"
HOSP_DIR = os.path.join(MIMIC_DIR, "hosp")  # change if your CSVs are elsewhere
ED_DIR = os.path.join(MIMIC_DIR, "ed")
LOOKBACK_DAYS = 30
LOOKBACK_DAYS2 = (
    730 if DATA_VERSION in ["2025-10-09-v3", "2025-10-09-v4", "2025-10-09-v5"] else 180
)
MAX_IDX_GAP = 9999
PREDICTION_GAP = 1 if DATA_VERSION in ["2025-10-09-v2", "2025-10-09-v3"] else 0


def get_lab_complete_flag(data_version: str) -> bool:
    """
    returns whether to apply lab-complete filtering.
    """
    if data_version in ["2025-10-09-v4"]:
        return True
    elif data_version in ["2025-10-09-v7"]:
        return False
    else:
        raise ValueError(f"Unknown DATA_VERSION for LAB_COMPLETE: {data_version}")


LAB_COMPLETE = get_lab_complete_flag(DATA_VERSION)


def get_lo_hi_from_idx(idx):
    if DATA_VERSION.startswith("2025-10-09"):
        lo = idx - pd.Timedelta(days=LOOKBACK_DAYS2)
        hi = idx - pd.Timedelta(days=PREDICTION_GAP)
    else:
        lo = idx - pd.Timedelta(days=LOOKBACK_DAYS)
        hi = idx + pd.Timedelta(days=LOOKBACK_DAYS)
    return lo, hi


LAB_ITEMS = {  # MIMIC itemid → short name
    51222: "HGB",  # Hemoglobin (혈색소)
    51250: "MCV",  # Mean Corpuscular Volume (평균 적혈구 용적)
    50882: "Bicarbonate",
    50902: "Chloride",
    50971: "Potassium",
    50983: "Sodium",
    51006: "Urea_Nitrogen",
    50912: "Creatinine",
    50931: "Glucose",
    50868: "Anion_Gap",
    51279: "Red_Blood_Cells",
    51301: "White_Blood_Cells",
    51221: "Hematocrit",
    51249: "MCHC",
    51265: "Platelet_Count",
    51277: "RDW",
    51248: "MCH",
    51146: "Basophils",
    51200: "Eosinophils",
    51244: "Lymphocytes",
    51254: "Monocytes",
    51256: "Neutrophils",
}


SELECT_CANCER = "C"  # C: Colorectal, L: Liver, E: Esophageal, P: Pancreatic


def load_preprocessed(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# ---- columns corresponding to lab features (used for complete-case filtering) ----
LAB_COLS = list(LAB_ITEMS.values())


class MyCounter:
    def __init__(self):
        self.n_adm_case = 0
        self.n_case = 0
        self.n_adm_ctrl = 0
        self.n_ctrl = 0


def compute_split_time(df):

    df = df.dropna(subset=["c_time"]).copy()
    df["c_time"] = pd.to_datetime(df["c_time"])
    df = df.sort_values("c_time").reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise RuntimeError("GC cases not found (after lab-complete filtering).")

    idx2 = max(0, math.floor(0.85 * n) - 1)
    split_time2 = df["c_time"].iloc[idx2]

    # 3) split_time1: split_time2 이하 집합에서 다시 85% 순서통계
    gc_pre = df[df["c_time"] <= split_time2].reset_index(drop=True)
    m = len(gc_pre)
    idx1 = max(0, math.floor(0.85 * m) - 1)
    split_time1 = gc_pre["c_time"].iloc[idx1]

    return split_time1, split_time2


def extract_feature(codes, versions, code_dict):
    codes = list(codes)
    versions = list(versions)
    # codes, versions는 같은 길이의 리스트
    for code, ver in zip(codes, versions):
        if ver == 9 and any(code.startswith(x) for x in code_dict["icd9"]):
            return 1
        elif ver == 10 and any(code.startswith(x) for x in code_dict["icd10"]):
            return 1
    return 0


def make_features_core(
    row,
    diag_small,
    lab_small,
    *,
    pat_df,
    var_dict,
    lab_items,
    match_date,
    prediction_gap,
    min_admittime=None,
    my_counter: Optional[MyCounter] = None,
):
    assert USE_LAB
    """Side-effect free feature builder (no nonlocal counters).
    Uses earliest lab values when DATA_VERSION == '2025-10-09'."""
    sid, idx, label = row.subject_id, row.c_time, row.label
    if label == 1:
        assert pd.notna(idx)

    # diagnoses up to index (for cases) or all (for controls when not matching by date)
    if label == 1 or match_date or DATA_VERSION == "2025-10-09-v6":
        hi = idx - pd.Timedelta(days=prediction_gap)
        diag_filtered = diag_small[
            (diag_small.subject_id == sid) & (diag_small.admittime <= hi)
        ]
        if DATA_VERSION in ["2025-10-09-v5", "2025-10-09-v6"]:
            lo = idx - pd.Timedelta(days=LOOKBACK_DAYS2)
            diag_filtered = diag_filtered[diag_filtered.admittime >= lo]
    else:
        diag_filtered = diag_small[(diag_small.subject_id == sid)]

    if label == 1:
        if my_counter is not None:
            my_counter.n_adm_case += diag_filtered.hadm_id.nunique()
            my_counter.n_case += 1
    else:
        if my_counter is not None:
            my_counter.n_adm_ctrl += diag_filtered.hadm_id.nunique()
            my_counter.n_ctrl += 1

    # basic features
    feats = {
        "AGE": pat_df.loc[pat_df.subject_id == sid, "anchor_age"].iat[0],
        "MALE": int(pat_df.loc[pat_df.subject_id == sid, "gender"].iat[0] == "M"),
    }

    # ICD-derived binary indicators
    codes = diag_filtered.icd_code_up
    versions = diag_filtered.icd_version

    for k, code_dict in var_dict.items():
        feats[k] = extract_feature(codes, versions, code_dict)

    # admittime gap (only meaningful if matching by date)
    if (
        match_date
        and pd.notna(row.c_time)
        and (min_admittime is not None)
        and pd.notna(min_admittime)
    ):
        feats["admittime_GAP"] = (row.c_time - min_admittime).total_seconds() / 86400.0
    else:
        feats["admittime_GAP"] = np.nan

    # Lab features
    if lab_small is not None:
        if label == 1 or DATA_VERSION == "2025-10-09-v6":
            lo, hi = get_lo_hi_from_idx(idx)
            l = lab_small[
                (lab_small.subject_id == sid) & (lab_small.charttime.between(lo, hi))
            ]
        else:
            l = lab_small[lab_small.subject_id == sid]

        if not l.empty:
            # For 2025-10-09, pick the oldest (earliest) value per item; otherwise latest
            select_oldest = USE_OLDEST
            grouped = l.sort_values("charttime").groupby("itemid")
            chosen_vals = (
                (grouped.head(1) if select_oldest else grouped.tail(1))
                .set_index("itemid")
                .valuenum
            )
            for itemid, name in lab_items.items():
                feats[name] = chosen_vals.get(itemid, np.nan)
        else:
            for name in lab_items.values():
                feats[name] = np.nan

    return pd.Series(feats)


def run_preprocess(verbose=False):
    data_loaded = False
    if os.path.exists(PREPROCESSED_FILE):
        print(f"Loading preprocessed data from {PREPROCESSED_FILE}")
        loaded = load_preprocessed(PREPROCESSED_FILE)
        file_version = loaded["data_version"]
        if file_version == DATA_VERSION:
            controls_features_dict = loaded["controls_features_dict"]
            cases_features_dict = loaded["cases_features_dict"]
            data_MATCH_DATE = loaded["MATCH_DATE"]

            assert (
                data_MATCH_DATE == MATCH_DATE
            ), f"Mismatch in MATCH_DATE: file={data_MATCH_DATE} vs code={MATCH_DATE}"
            data_loaded = True
            print(f"Data version matches ({file_version}) → Loaded from file")
        else:
            print(
                f"Data version mismatch: file={file_version} vs code={DATA_VERSION}. Will regenerate."
            )
    # 추가
    if data_loaded:
        return controls_features_dict, cases_features_dict

    if not data_loaded:
        # ICD code lists -------------------------------------------------------------

        GC_CODES = {
            "icd9": [
                "1510",
                "1511",
                "1512",
                "1513",
                "1514",
                "1515",
                "1516",
                "1518",
                "1519",
            ],
            "icd10": [
                "C16",
                "C160",
                "C161",
                "C162",
                "C163",
                "C164",
                "C165",
                "C166",
                "C168",
                "C169",
            ],
        }

        COLORECTAL_CANCER_CODES = {  # 항문암은 제외, 정확히 대장암만 포함
            "icd9": [
                "1530",
                "1531",
                "1532",
                "1533",
                "1534",
                "1535",
                "1536",
                "1537",
                "1538",
                "1539",
                "1540",
                "1541",
            ],
            "icd10": [
                "C18",
                "C180",
                "C181",
                "C182",
                "C183",
                "C184",
                "C185",
                "C186",
                "C187",
                "C188",
                "C189",
                "C19",
                "C20",
            ],
        }

        LIVER_CANCER_CODES = {
            "icd9": ["1550", "1551", "1552"],
            "icd10": [
                "C22",
                "C220",
                "C221",
                "C222",
                "C223",
                "C224",
                "C227",
                "C228",
                "C229",
            ],
        }

        ESOPHAGEAL_CANCER_CODES = {
            "icd9": [
                "1500",
                "1501",
                "1502",
                "1503",
                "1504",
                "1505",
                "1508",
                "1509",
            ],
            "icd10": [
                "C15",
                "C153",
                "C154",
                "C155",
                "C158",
                "C159",
            ],
        }

        PANCREATIC_CANCER_CODES = {
            "icd9": ["1570", "1571", "1572", "1573", "1574", "1578", "1579"],
            "icd10": [
                "C25",
                "C250",
                "C251",
                "C252",
                "C253",
                "C254",
                "C257",
                "C258",
                "C259",
            ],
        }

        # 이 위로 icd코드 체크됨

        CODESETS = {
            "G": GC_CODES,
            "C": COLORECTAL_CANCER_CODES,
            "L": LIVER_CANCER_CODES,
            "E": ESOPHAGEAL_CANCER_CODES,
            "P": PANCREATIC_CANCER_CODES,
        }

        GI_NEG = set(
            [
                "K29",  # Gastritis and duodenitis (위염 및 십이지장염)
                "K25",  # Gastric ulcer (위궤양)
                "K26",  # Duodenal ulcer (십이지장 궤양)
                "K27",  # Peptic ulcer, site unspecified (소화성 궤양, 부위 미지정)
                "K28",  # Gastrojejunal ulcer (위공장 궤양)
                "K30",  # Functional dyspepsia (기능성 소화불량)
                "K21",  # Gastro-esophageal reflux disease (위식도 역류 질환)
                "R1013",  # Epigastric pain (상복부 통증)
            ]
        )
        VAR = {
            "DIABETES": {
                "icd9": ["249", "250", "6480"],
                "icd10": ["E08", "E09", "E10", "E11", "E13", "O24", "P702"],
            },
            "OBESITY": {"icd9": ["2780"], "icd10": ["E66"]},
            "ALCOHOL": {"icd9": ["303", "3050"], "icd10": ["F10"]},
            "SMOKE": {
                "icd9": ["3051", "V1582"],
                "icd10": ["F17", "Z720", "Z716", "Z87891"],
            },
            "GERD": {"icd9": ["53081"], "icd10": ["K21"]},
            "DYSPEPSIA": {"icd9": ["5368"], "icd10": ["K30"]},
            "ULCER": {
                "icd9": ["531", "532", "533", "534"],
                "icd10": ["K25", "K26", "K27", "K28"],
            },
            "GASTRITIS": {"icd9": ["535"], "icd10": ["K29"]},
            "HPYLORI": {"icd9": ["04186"], "icd10": ["B9681"]},
            "HYPERTENSION": {
                "icd9": ["401"],
                "icd10": ["I10", "I11", "I12", "I13", "I15"],
            },
            "MI_AP": {
                "icd9": ["410", "412", "413"],
                "icd10": ["I21", "I22", "I252", "I20"],
            },
            "DYSLIPIDEMIA": {"icd9": ["272"], "icd10": ["E78"]},
            "GA": {"icd9": ["2111"], "icd10": ["D131"]},  # Gastric adenoma
            "FAMILY_HISTORY_CANCER": {"icd9": ["V160"], "icd10": ["Z80"]},
            # 새로 추가된 항목
            "PROTEIN_CALORIE_MALNUTRITION": {
                "icd9": ["260", "261", "262", "263"],
                "icd10": ["E40", "E41", "E42", "E43", "E44", "E45", "E46"],
            },
            "NEOPLASM_INTRA_ABDOMINAL_LYMPH": {
                "icd9": ["20285"],
                "icd10": ["C772"],
            },
            "NEOPLASM_RETROPERITONEUM_PERITONEUM": {
                "icd9": ["158", "1590"],
                "icd10": ["C48"],
            },
            "POSTHEMORRHAGIC_ANEMIA": {
                "icd9": ["2851"],
                "icd10": ["D500"],
            },
            "ADVERSE_EFFECT_ANTINEOPLASTIC": {
                "icd9": ["E9331"],
                "icd10": ["T451X5A"],
            },
            "DYSPHAGIA": {
                "icd9": ["7872"],
                "icd10": ["R13"],
            },
            "ANTINEOPLASTIC_CHEMOTHERAPY": {
                "icd9": ["V5811"],
                "icd10": ["Z5111"],
            },
            "ATRIAL_FIBRILLATION": {
                "icd9": ["42731"],
                "icd10": ["I480", "I481", "I482", "I4891"],
            },
            "IRON_DEFICIENCY_ANEMIA": {
                "icd9": ["280"],
                "icd10": ["D50"],
            },
            "KIDNEY_FAILURE": {
                "icd9": ["585", "586"],
                "icd10": ["N17", "N18", "N19"],
            },
            "CORONARY_ATHEROSCLEROSIS_NATIVE": {
                "icd9": ["41401"],
                "icd10": ["I2510", "I2511"],
            },
            "ATHEROSCLEROTIC_HEART_DISEASE_NATIVE_NO_ANGINA": {
                "icd9": ["41401"],
                "icd10": ["I2510"],
            },
            "PURE_HYPERCHOLESTEROLEMIA": {
                "icd9": ["2720"],
                "icd10": ["E780"],
            },
            "HYPO_OSMOLALITY_HYPONATREMIA": {
                "icd9": ["2761"],
                "icd10": ["E871"],
            },
        }

        # -------------- 1.  Read core tables --------------------
        pat = pd.read_csv(
            os.path.join(HOSP_DIR, "patients.csv.gz"),
            usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        )
        adm = pd.read_csv(
            os.path.join(HOSP_DIR, "admissions.csv.gz"),
            usecols=["subject_id", "hadm_id", "admittime", "dischtime", "race"],
        )

        # %%
        # admissions에서 환자별 첫 번째 race 값을 patients 테이블과 병합
        race_df = (
            adm[["subject_id", "race"]]
            .dropna(subset=["race"])  # race가 기록된 경우만 (race 값이 없는 행 drop)
            .drop_duplicates("subject_id")  # subject_id 당 한 행
        )
        pat = pat.merge(race_df, on="subject_id", how="left")

        # %%

        # race가 비어‑있는 환자는 'UNKNOWN'으로 채움
        pat["race"] = pat["race"].fillna("UNKNOWN")

        # %%

        diag = pd.read_csv(
            os.path.join(HOSP_DIR, "diagnoses_icd.csv.gz"),
            usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        )

        if USE_LAB:
            lab = pd.read_csv(
                os.path.join(HOSP_DIR, "labevents.csv.gz"),
                usecols=[
                    "subject_id",
                    "hadm_id",
                    "itemid",
                    "charttime",
                    "value",
                    "valuenum",
                    "valueuom",
                ],
            )
            # lab 테이블의 row를 필터링(lab.itemid를 LAB_ITEMS의 key로 제한)
            lab = lab[lab.itemid.isin(LAB_ITEMS.keys())]

        # convert times --------------------------------------------------------------
        adm["admittime"] = pd.to_datetime(adm.admittime)
        diag = diag.merge(
            adm[["hadm_id", "admittime"]], on="hadm_id", how="left"
        )  # diag테이블에 admittime 열 추가

        # %%
        # -------------- 2. Identify first GC diagnosis ------------------------------
        diag["icd_code_up"] = diag.icd_code.str.upper().str.replace(
            ".", "", regex=False
        )

        def subjects_in_window(t1, t2):
            return set(
                adm.loc[
                    (adm["admittime"] > t1) & (adm["admittime"] <= t2), "subject_id"
                ].unique()
            )

        def split_case_dict(
            case_df: pd.DataFrame, sp1, sp2
        ):  # 단일 case 데이터프레임 들어왔을 때 return [train/val/test]

            split_time1, split_time2 = sp1, sp2
            loc = case_df.copy()
            if "c_time" not in loc.columns:
                raise KeyError("no 'c_time' column")
            if not pd.api.types.is_datetime64_any_dtype(loc["c_time"]):
                loc["c_time"] = pd.to_datetime(loc["c_time"], errors="coerce")
            loc = loc.dropna(subset=["c_time"])
            m_train = loc["c_time"] <= split_time1
            m_val = (loc["c_time"] > split_time1) & (loc["c_time"] <= split_time2)
            m_test = loc["c_time"] > split_time2
            return [
                loc[m_train].reset_index(drop=True),
                loc[m_val].reset_index(drop=True),
                loc[m_test].reset_index(drop=True),
            ]

        def has_icd_code(codes, version, code_dict):
            """
            codes: set of code strings for a patient/admission
            version: ICD version (9 or 10)
            code_dict: dictionary {"icd9": [...], "icd10": [...]}
            """
            if version == 9:
                return any(c.startswith(tuple(code_dict["icd9"])) for c in codes)
            elif version == 10:
                return any(c.startswith(tuple(code_dict["icd10"])) for c in codes)
            else:
                return False

        # %%
        def make_cases_df(diag_df: pd.DataFrame, code_dict: dict):
            """
            주어진 ICD 코드 사전(code_dict)에 대해
            - 환자별 '첫 진단 시점'을 구하고
            - 가장 늦은 진단 시점 + max_idx 계산
            - (subject_id, c_time, label=1) 형태의 cases DataFrame을 반환

            Parameters
            ----------
            diag_df : pd.DataFrame
                컬럼에 ['subject_id','admittime','icd_code_up','icd_version'] 포함
            code_dict : dict
                {"icd9": [...], "icd10": [...]} prefix 리스트
                저장할 시점 컬럼명 (예: 'c_time', 'crc_time' 등)
            max_idx_gap_days : int
                max_idx 계산 시 더할 일 수

            Returns
            -------
            first_dx : pd.DataFrame
                환자별 첫 진단 행만 가진 표. 컬럼은 ['subject_id', c_time] 포함
            cases : pd.DataFrame
                ['subject_id', 'c_time, 'label'] (label=1) 형태
            """
            # 행 단위로 이 암에 해당하는지 여부 계산
            is_dx = diag_df.progress_apply(
                lambda r: has_icd_code([r.icd_code_up], r.icd_version, code_dict),
                axis=1,
            )

            # 환자별 가장 이른(admittime) 진단만 남기기
            first_dx = (  #
                diag_df[is_dx]
                .sort_values("admittime")
                .groupby("subject_id")
                .first()
                .reset_index()
                .loc[:, ["subject_id", "admittime"]]
                .rename(
                    columns={"admittime": "c_time"}
                )  # admittime 열을 time_col로 변경
            )

            # 학습용 case 테이블(label=1)
            cases = first_dx.copy()
            cases["label"] = 1

            return cases  # case table 반환

        # %%

        case_dict_base = {}
        for k, v in CODESETS.items():
            case_dict_base[k] = make_cases_df(diag_df=diag, code_dict=v)

        # %%
        # data 파일 없음
        if USE_ED:
            diag_ed = pd.read_csv(os.path.join(ED_DIR, "diagnosis.csv.gz"))
            print(diag_ed.shape)
            print(diag.shape)
            dia_s_set = set(diag.subject_id)
            diag_ed_s_set = set(diag_ed.subject_id)
            print(len(diag_ed_s_set - dia_s_set))

        ############################
        # %%
        # ---------------- 3‑A. Identify potential controls' subject_id -----------------------

        # --- 1. 암 진단 환자 추출 --------------------------------------------

        # gc_subj = set(cases_dict["G"].subject_id.unique())
        # colorectal_subj = set(cases_dict["C"].subject_id.unique())
        # liver_subj = set(cases_dict["L"].subject_id.unique())
        # esophageal_subj = set(cases_dict["E"].subject_id.unique())
        # pancreatic_subj = set(cases_dict["P"].subject_id.unique())

        # --- 2. controls_pool 수정 ------------------------------------------

        # print("---------------------------------------------------")
        # print(f"length of gc_subjects: {len(gc_subj)}")
        # print(f"length of colorectal cancer subjects: {len(colorectal_subj)}")
        # print(f"length of liver cancer subjects: {len(liver_subj)}")
        # print(f"length of esophageal cancer subjects: {len(esophageal_subj)}")
        # print(f"length of pancreatic cancer subjects: {len(pancreatic_subj)}")

        # ======================================================================
        # SPECIAL PIPELINE for DATA_VERSION == "2025-10-09"
        # Build datasets using ONLY subjects with NO missing Lab features,
        # while keeping the original case:control ratio and avoiding expensive
        # all-controls lab feature extraction.
        # Assumptions per request:
        # - MATCH_DATE is assumed False (no date-matching logic required)
        # - No stratified sampling for controls (pure random)
        # - Build case lab features first, filter to lab-complete
        # - Then iteratively sample controls in batches: compute features, keep
        #   only lab-complete, and stop once target count is met.
        # ======================================================================
        if DATA_VERSION.startswith("2025-10-09") and USE_LAB and not MATCH_DATE:
            rng = np.random.RandomState(g_seed)
            lab_cols = list(LAB_ITEMS.values())

            def is_lab_complete(df):
                if not lab_cols or not LAB_COMPLETE:
                    return pd.Series([True] * len(df), index=df.index)
                return df[lab_cols].notna().all(axis=1)

            cases_features_dict = {}
            controls_features_dict = {}
            if TIME_SPLIT:
                cases_features_dict_train = {}
                cases_features_dict_val = {}
                cases_features_dict_test = {}
                cases_features_dict["train"] = cases_features_dict_train
                cases_features_dict["val"] = cases_features_dict_val
                cases_features_dict["test"] = cases_features_dict_test

                controls_features_dict_train = {}
                controls_features_dict_val = {}
                controls_features_dict_test = {}
                controls_features_dict["train"] = controls_features_dict_train
                controls_features_dict["val"] = controls_features_dict_val
                controls_features_dict["test"] = controls_features_dict_test
            # ---- Per cancer type (keys in CODESETS): build cases first, then controls ----
            for k, cases_df in case_dict_base.items():
                # 1) Build case features and keep only lab-complete
                diag_cases_small = diag[
                    diag.subject_id.isin(set(cases_df.subject_id))
                ].copy()

                if USE_LAB:
                    lab_cases_small = lab[
                        lab.subject_id.isin(set(cases_df.subject_id))
                        & lab.itemid.isin(LAB_ITEMS.keys())
                    ].copy()
                    if not lab_cases_small.empty:
                        lab_cases_small["charttime"] = pd.to_datetime(
                            lab_cases_small["charttime"]
                        )
                else:
                    lab_cases_small = None

                cases_feat = pd.concat(
                    [
                        cases_df,
                        cases_df.progress_apply(
                            lambda r: make_features_core(
                                r,
                                diag_cases_small,
                                lab_cases_small,
                                pat_df=pat,
                                var_dict=VAR,
                                lab_items=LAB_ITEMS,
                                match_date=MATCH_DATE,
                                prediction_gap=PREDICTION_GAP,
                            ),
                            axis=1,
                        ),
                    ],
                    axis=1,
                )

                cases_feat = cases_feat[is_lab_complete(cases_feat)].reset_index(
                    drop=True
                )
                print(
                    f"[{k}] Initial cases: {len(cases_df)} → lab-complete: {len(cases_feat)}"
                )
                if TIME_SPLIT:
                    split_time1, split_time2 = compute_split_time(cases_feat)
                    ls = split_case_dict(cases_feat, split_time1, split_time2)
                    cases_features_dict["train"][k] = ls[0]
                    cases_features_dict["val"][k] = ls[1]
                    cases_features_dict["test"][k] = ls[2]

                    # 2) Determine target #controls from remaining lab-complete cases
                    n_train = len(cases_features_dict["train"][k])
                    n_val = len(cases_features_dict["val"][k])
                    n_test = len(cases_features_dict["test"][k])

                    target_controls_dict = {
                        "train": int(n_train * CONTROL_RATIO),
                        "val": int(n_val * CONTROL_RATIO),
                        "test": int(n_test * CONTROL_RATIO),
                    }
                else:
                    cases_features_dict[k] = cases_feat
                    target_controls = int(len(cases_feat) * CONTROL_RATIO)

                # 3) Prepare control pool (exclude subjects who are cases of THIS cancer type)
                excluded_subjects = set(cases_df.subject_id.unique())
                control_pool = set(pat.subject_id.unique()) - excluded_subjects

                # Helper to sample a batch of controls and compute lab-complete features
                def sample_control_batch(
                    pool_ids, n_batch, attempt_idx, t1=None, t2=None
                ):
                    if TIME_SPLIT:
                        elig = pool_ids & subjects_in_window(t1, t2)
                        n_batch = min(n_batch, len(elig))
                    else:
                        n_batch = min(n_batch, len(pool_ids))
                    if n_batch <= 0:
                        return pd.DataFrame(), set()

                    if TIME_SPLIT:
                        chosen = rng.choice(list(elig), size=n_batch, replace=False)
                    else:
                        chosen = rng.choice(list(pool_ids), size=n_batch, replace=False)
                    chosen_set = set(chosen)

                    # Build minimal "controls" rows: label=0, c_time is NaT (unused when MATCH_DATE=False)
                    ctrl_rows = pd.DataFrame(
                        {
                            "subject_id": list(chosen_set),
                            "label": 0,
                            "c_time": pd.NaT,
                        }
                    )

                    # Filter big tables down to the chosen subjects for speed
                    diag_ctrl_small = diag[diag.subject_id.isin(chosen_set)].copy()
                    if TIME_SPLIT:
                        diag_ctrl_small = diag_ctrl_small[
                            (diag_ctrl_small.admittime > t1)
                            & (diag_ctrl_small.admittime <= t2)
                        ]

                    if USE_LAB:
                        lab_ctrl_small = lab[
                            lab.subject_id.isin(chosen_set)
                            & lab.itemid.isin(LAB_ITEMS.keys())
                        ].copy()
                        if not lab_ctrl_small.empty:
                            lab_ctrl_small["charttime"] = pd.to_datetime(
                                lab_ctrl_small["charttime"]
                            )
                            if TIME_SPLIT:
                                lab_ctrl_small = lab_ctrl_small[
                                    (lab_ctrl_small["charttime"] > t1)
                                    & (lab_ctrl_small["charttime"] <= t2)
                                ]

                    else:
                        lab_ctrl_small = None

                    ctrl_feat = pd.concat(
                        [
                            ctrl_rows,
                            ctrl_rows.progress_apply(
                                lambda r: make_features_core(
                                    r,
                                    diag_ctrl_small,
                                    lab_ctrl_small,
                                    pat_df=pat,
                                    var_dict=VAR,
                                    lab_items=LAB_ITEMS,
                                    match_date=MATCH_DATE,
                                    prediction_gap=PREDICTION_GAP,
                                ),
                                axis=1,
                            ),
                        ],
                        axis=1,
                    )

                    print(
                        f"[{k}] Initial controls: {len(ctrl_feat)} → lab-complete: {is_lab_complete(ctrl_feat).sum()}"
                    )
                    ctrl_feat = ctrl_feat[is_lab_complete(ctrl_feat)].reset_index(
                        drop=True
                    )
                    return ctrl_feat, chosen_set

                # 4) Iteratively sample batches until we reach the target
                def make_controls_features_dict(
                    target_controls: int, controls_pool: set, t1=None, t2=None
                ):
                    remaining = target_controls
                    accumulated = []  # list of lab-complete control feature dataframes
                    attempt = 0

                    while remaining > 0 and len(controls_pool) > 0:
                        attempt += 1
                        # Start with 2x the remaining target (as specified)
                        n_controls = max(1, 2 * remaining)
                        n_controls = min(n_controls, len(controls_pool))

                        batch_feat, sampled_ids = sample_control_batch(
                            controls_pool, n_controls, attempt, t1, t2
                        )

                        # Remove sampled ids from pool (regardless of lab-complete or not)
                        controls_pool -= sampled_ids

                        if batch_feat.empty:
                            # No lab-complete controls in this batch; continue
                            continue

                        if len(batch_feat) >= remaining:
                            # Take only what's needed and stop
                            accumulated.append(
                                batch_feat.sample(
                                    n=remaining, replace=False, random_state=g_seed
                                )
                            )
                            remaining = 0
                            break
                        else:
                            # Take all and continue
                            accumulated.append(batch_feat)
                            remaining -= len(batch_feat)
                    # assert (
                    #     len(accumulated) >= 1
                    # ), "Should have at least some lab-complete controls by now"
                    if accumulated:
                        controls_feat_final = pd.concat(accumulated, ignore_index=True)
                    else:
                        controls_feat_final = pd.DataFrame(
                            columns=list(cases_feat.columns)
                        )

                    # Enforce exact target size if we somehow exceeded (safety)
                    if len(controls_feat_final) > target_controls:
                        controls_feat_final = controls_feat_final.sample(
                            n=target_controls, replace=False, random_state=g_seed
                        ).reset_index(drop=True)

                    return controls_feat_final.reset_index(drop=True), controls_pool

                if TIME_SPLIT:
                    for key, value in target_controls_dict.items():
                        if key == "train":
                            t1, t2 = pd.Timestamp.min, split_time1
                        elif key == "val":
                            t1, t2 = split_time1, split_time2
                        else:  # 'test'
                            t1, t2 = split_time2, pd.Timestamp.max

                        controls_feat_k, control_pool = make_controls_features_dict(
                            value, control_pool, t1, t2
                        )
                        controls_features_dict[key][k] = controls_feat_k
                else:
                    controls_features_dict[k] = make_controls_features_dict(
                        target_controls=target_controls, controls_pool=control_pool
                    )[0]
            # Save and return early
            to_save = {
                "data_version": DATA_VERSION,
                "MATCH_DATE": MATCH_DATE,
                "controls_features_dict": controls_features_dict,
                "cases_features_dict": cases_features_dict,
            }
            with open(PREPROCESSED_FILE, "wb") as f:
                pickle.dump(to_save, f)
            print(
                f"[SPECIAL 2025-10-09] Saved preprocessed data to {PREPROCESSED_FILE} "
                f"with version {DATA_VERSION} (MATCH_DATE={MATCH_DATE})"
            )

            if verbose:
                if TIME_SPLIT:
                    for k, v in cases_features_dict.items():
                        for key, value in cases_features_dict[k].items():
                            count_nan(value)
                    for k, v in controls_features_dict.items():
                        for key, value in controls_features_dict[k].items():
                            count_nan(value)
                else:
                    for k, v in cases_features_dict.items():
                        count_nan(v)
                    for k, v in controls_features_dict.items():
                        count_nan(v)
                print("====================================================")

            return controls_features_dict, cases_features_dict
        # ======================================================================
        # END SPECIAL PIPELINE
        # ======================================================================

        # %%
        # ---------------- 3‑BIS. Robust control‑matching -----------------

        def match_controls(case_row, controls_pool):
            """
            Returns a DataFrame with exactly CONTROL_RATIO matched controls.
            """

            sid = case_row["subject_id"]
            idx_date = case_row["c_time"]  # 추가: case의 c_time을 idx_date로 사용

            hi = idx_date - pd.Timedelta(
                days=PREDICTION_GAP
            )  # 추가: controls뽑을 때 hi이전 시점의 진단 기록을 가지고 있는지 확인용

            masks = []
            if MATCH_DATE:
                left_cond = idx_date - pd.Timedelta(days=DAYS_GAP) <= diag.admittime
                right_cond = diag.admittime <= idx_date + pd.Timedelta(days=DAYS_GAP)
                diag_mask = left_cond & right_cond
                diag_filtered = diag[diag_mask].copy()
                applicable = set(diag[diag_mask].subject_id)
                base2 = pat.subject_id.isin(applicable)
                masks.append(base2)
            age, sex, race = pat.loc[
                pat.subject_id == sid, ["anchor_age", "gender", "race"]
            ].values[0]
            # --- Step‑wise masks ----------------------------------------------------

            # 1. strict
            base = pat.subject_id.isin(controls_pool)
            masks.append(base)

            # --- try the masks in order --------------------------------------------
            if MATCH_DATE:
                pool = pat[masks[0] & masks[1]].subject_id
            else:
                pool = pat[masks[0]].subject_id
            n = len(pool)
            if n >= CONTROL_RATIO:  # CONTROL_RATIO 개수만큼 무작위로 중복 없이 추출
                chosen = pool.sample(CONTROL_RATIO, replace=False, random_state=g_seed)

            # 모자란 수(CONTROL_RATIO - n)만큼 중복 허용하여 추가 샘플링
            elif 0 < n < CONTROL_RATIO:
                # not enough – take all without replacement, rest with replacement
                extra = pool.sample(
                    CONTROL_RATIO - n, replace=True, random_state=g_seed
                )
                chosen = pd.concat([pool, extra])
            else:  # n==0일 경우
                # no candidate at all (rare) – skip this case
                # 위 if 조건에 한 번도 해당 안 된 경우
                return pd.DataFrame(columns=["subject_id", "label", "c_time"])
            controls_pool -= set(chosen.values)  # remove chosen controls from pool

            if MATCH_DATE:
                chosen_id_set = set(chosen.values)
                diag_filtered = diag_filtered[
                    diag_filtered.subject_id.isin(chosen_id_set)
                ]
                sampled_c_time = []
                for subject_id in chosen.values:
                    sampled_c_time.append(
                        diag_filtered[diag_filtered.subject_id == subject_id]
                        .admittime.sample(1, random_state=g_seed)
                        .values[0]
                    )

            else:
                sampled_c_time = None
            return pd.DataFrame(
                {
                    "subject_id": chosen.values,
                    "label": 0,
                    "c_time": sampled_c_time,
                }  # 변경: c_time을 idx_date로 설정
            )

        # %%
        # ctrl_list = [match_controls(r) for _, r in cases.iterrows()]
        # ctrl_list = [df for df in ctrl_list if not df.empty]
        # controls = pd.concat(ctrl_list, ignore_index=True)

        controls_features_dict = {}
        for k, v in cases_features_dict.items():
            excluded_subjects = set(cases_features_dict[k].subject_id.unique())
            controls_pool = set(pat.subject_id.unique()) - excluded_subjects
            ctrl_list = [match_controls(r, controls_pool) for _, r in v.iterrows()]
            ctrl_list = [df for df in ctrl_list if not df.empty]
            controls_features_dict[k] = pd.concat(ctrl_list, ignore_index=True)

        # %%
        cohort_dict = {}
        for k in cases_features_dict.keys():
            cohort_dict[k] = pd.concat(
                [cases_features_dict[k], controls_features_dict[k]], ignore_index=True
            )
        # %%
        if USE_LAB:  # lab_small_dict 만들기
            lab_small_dict = (
                {}
            )  # lab_small: lab에서 subject_id가 cohort의 subject_id에 포함됨 & itemid가 LAB_ITEMS의 itemid에 포함됨
            for k, v in cohort_dict.items():
                lab_small_dict[k] = lab[
                    lab.subject_id.isin(set(v.subject_id))
                    & lab.itemid.isin(LAB_ITEMS.keys())
                ].copy()
                lab_small_dict[k]["charttime"] = pd.to_datetime(
                    lab_small_dict[k].charttime
                )
        diag_small_dict = (
            {}
        )  # diag_small: diag에서 subject_id가 cohort의 subject아이디에 포함됨
        for k, v in cohort_dict.items():
            diag_small_dict[k] = diag[diag.subject_id.isin(set(v.subject_id))].copy()

        # %%
        # -------------- 4. Feature engineering --------------------------------------

        min_c_time_case = cases_features_dict["G"]["c_time"].min()
        min_c_time_ctrl = (
            controls_features_dict["G"]["c_time"].min()
            if "c_time" in controls_features_dict["G"].columns
            else pd.NaT
        )
        min_admittime = min(
            [d for d in [min_c_time_case, min_c_time_ctrl] if pd.notna(d)]
        )

        my_counter = MyCounter()

        def make_features(row, diag_small, lab_small):
            assert MATCH_DATE is False
            sid, idx, label = row.subject_id, row.c_time, row.label
            # lo = idx - pd.Timedelta(days=LOOKBACK_DAYS)

            if label == 1 or MATCH_DATE:
                hi = idx - pd.Timedelta(days=PREDICTION_GAP)
                diag_filtered = diag_small[
                    (diag_small.subject_id == sid) & (diag_small.admittime <= hi)
                ]
            else:
                diag_filtered = diag_small[(diag_small.subject_id == sid)]

            if label == 1:
                if my_counter is not None:
                    my_counter.n_adm_case += diag_filtered.hadm_id.nunique()
                    my_counter.n_case += 1
            else:
                if my_counter is not None:
                    my_counter.n_adm_ctrl += diag_filtered.hadm_id.nunique()
                    my_counter.n_ctrl += 1
            codes = diag_filtered.icd_code_up
            versions = diag_filtered.icd_version
            # ICD features -----------------------------------------------------------
            feats = {
                "AGE": pat.loc[pat.subject_id == sid, "anchor_age"].iat[0],
                "MALE": int(pat.loc[pat.subject_id == sid, "gender"].iat[0] == "M"),
            }

            for k, code_dict in VAR.items():
                feats[k] = extract_feature(codes, versions, code_dict)

            if MATCH_DATE and pd.notna(row.c_time) and pd.notna(min_admittime):
                feats["admittime_GAP"] = (
                    row.c_time - min_admittime
                ).total_seconds() / 86400.0
            else:
                feats["admittime_GAP"] = np.nan

            # Lab features -----------------------------------------------------------
            # subject_id_set을 지우며 임시 주석처리함

            if USE_LAB:
                if (
                    label == 1
                ):  # 각 itemid 별로 유효기간(전후 30일)내 마지막 검사 결과 적용
                    lo, hi_lab = get_lo_hi_from_idx(idx)
                    l = lab_small[  # l: input으로 들어온 하나의 환자 & index date 전후 30일 검사로 필터링
                        (lab_small.subject_id == sid)
                        & (lab_small.charttime.between(lo, hi_lab))
                    ]
                else:
                    l = lab_small[lab_small.subject_id == sid]
                if not l.empty:
                    select_oldest = USE_OLDEST
                    grouped = l.sort_values("charttime").groupby("itemid")
                    chosen_vals = (
                        (grouped.head(1) if select_oldest else grouped.tail(1))
                        .set_index("itemid")
                        .valuenum
                    )

                    for itemid, name in LAB_ITEMS.items():
                        feats[name] = chosen_vals.get(itemid, np.nan)
                else:
                    for name in LAB_ITEMS.values():
                        feats[name] = np.nan

            return pd.Series(feats)

        # %%
        # cases, controls, cohort 의 shape를 출력하여 확인
        print("---------------------------------------------------")
        for k, v in cases_features_dict.items():
            print(f"cases_dict[{k}].shape = {v.shape}")
        print("---------------------------------------------------")
        for k, v in controls_features_dict.items():
            print(f"controls_dict[{k}].shape = {v.shape}")
            # print("length of unique controls subject id =", len(v.subject_id.unique()))
        print("---------------------------------------------------")
        for k, v in cohort_dict.items():
            print(f"cohort_dict[{k}].shape = {v.shape}")
        # %%
        cases_features_dict = {}
        for k, v in cases_features_dict.items():
            if USE_LAB:
                lab_small_dict_arg = lab_small_dict[k]
            else:
                lab_small_dict_arg = None
            cases_features_dict[k] = pd.concat(
                [
                    v,
                    v.progress_apply(
                        lambda r: make_features_core(
                            r,
                            diag_small_dict[k],
                            lab_small_dict_arg,
                            pat_df=pat,
                            var_dict=VAR,
                            lab_items=LAB_ITEMS,
                            match_date=MATCH_DATE,
                            prediction_gap=PREDICTION_GAP,
                            my_counter=my_counter,
                        ),
                        # make_features(
                        #     r, diag_small_dict[k], lab_small_dict_arg
                        # ),
                        axis=1,
                    ),
                ],
                axis=1,
            )

        controls_features_dict = {}

        for k, v in controls_features_dict.items():
            if USE_LAB:
                lab_small_dict_arg = lab_small_dict[k]
            else:
                lab_small_dict_arg = None
            controls_features_dict[k] = pd.concat(
                [
                    v,
                    v.progress_apply(
                        lambda r: make_features_core(
                            r,
                            diag_small_dict[k],
                            lab_small_dict_arg,
                            pat_df=pat,
                            var_dict=VAR,
                            lab_items=LAB_ITEMS,
                            match_date=MATCH_DATE,
                            prediction_gap=PREDICTION_GAP,
                            my_counter=my_counter,
                        ),
                        # make_features(
                        #     r, diag_small_dict[k], lab_small_dict_arg
                        # ),
                        axis=1,
                    ),
                ],
                axis=1,
            )
        # %%
        print("adm average count of case: ", my_counter.n_adm_case / my_counter.n_case)
        print(
            "adm average count of control: ", my_counter.n_adm_ctrl / my_counter.n_ctrl
        )

        # %%
        # for k, v in cases_features_dict.items():
        #     nan_imputation(v, LAB_ITEMS=LAB_ITEMS, USE_LAB=USE_LAB)

        # for k, v in controls_features_dict.items():
        #     nan_imputation(v, LAB_ITEMS=LAB_ITEMS, USE_LAB=USE_LAB)

        # %%

        # 저장 시
        to_save = {
            "data_version": DATA_VERSION,
            "MATCH_DATE": MATCH_DATE,
            "controls_features_dict": controls_features_dict,  # 대조군용 feature DataFrame
            "cases_features_dict": cases_features_dict,  # 암 환자용 feature DataFrame
        }
        with open(PREPROCESSED_FILE, "wb") as f:
            pickle.dump(to_save, f)
        print(
            f"Saved preprocessed data to {PREPROCESSED_FILE} with version {DATA_VERSION} (MATCH_DATE={MATCH_DATE})"
        )
    if verbose:
        for k, v in cases_features_dict.items():
            count_nan(v)
        for k, v in controls_features_dict.items():
            count_nan(v)
        print("====================================================")

    return controls_features_dict, cases_features_dict
