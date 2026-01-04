from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from nan import nan_imputation
from predcancer.preprocess import LAB_ITEMS, run_preprocess
from predcancer.settings import (FIX_TEST_SPLIT, MATCH_DATE,
                                 SAVE_CASE_CONTROL_STAT, SAVE_SCALER, USE_LAB,
                                 g_seed, ignore_cols, meta_config, num_cols)
from predcancer.utils import save_scaler
from show_stat import save_stat


def normalize(df: pd.DataFrame, other_df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Normalize the specified columns of a DataFrame using StandardScaler.

    Args:
        df (pd.DataFrame): The DataFrame to normalize.
        cols (list): List of column names to normalize.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """

    cols = list(num_cols)
    if USE_LAB:
        cols = cols + list(LAB_ITEMS.values())
    scaler = StandardScaler().fit(df[cols])

    df.loc[:, cols] = scaler.transform(df[cols])
    for other_df in other_df_list:
        other_df.loc[:, cols] = scaler.transform(other_df[cols])

    return scaler


def prepare_datasets(
    split_seed=g_seed,
    controls_features_dict=None,
    cases_features_dict=None,
    verbose=False,
):
    assert isinstance(verbose, bool), "verbose must be a boolean"

    if controls_features_dict is None or cases_features_dict is None:
        controls_features_dict, cases_features_dict = run_preprocess()

    for k, v in cases_features_dict.items():
        print(f"암종 {k}의 cases shape: {cases_features_dict[k].shape}")
        print(f"암종 {k}의 controls shape: {controls_features_dict[k].shape}")

    print("-------------------------------------------------------")

    # full_gc = pd.concat([cases_features_dict["G"], controls_features_dict["G"]], axis=0)
    # full_lc = pd.concat([cases_features_dict["L"], controls_features_dict["L"]], axis=0)

    # %%

    # "G": GC_CODES,
    # "C": COLORECTAL_CANCER_CODES, 대장암
    # "L": LIVER_CANCER_CODES,
    # "E": ESOPHAGEAL_CANCER_CODES, 식도암
    # "P": PANCREATIC_CANCER_CODES,

    # cases_features_dict: 각 암에 대한 cases 데이터프레임 dict
    # controls_features_dict: 각 암 환자에 대해 매칭한 controls 데이터프레임 dict
    # 암의 첫 대문자를 key로 접근("G", "C", "L", "E", "P")
    # %%
    def drop_features(df):
        drop_cols = [
            "HGB",
            "MCV",
            "Bicarbonate",
            "Chloride",
            "Potassium",
            "Sodium",
            "Urea_Nitrogen",
            "Creatinine",
            "Glucose",
            "Anion_Gap",
            "Red_Blood_Cells",
            "White_Blood_Cells",
            "Hematocrit",
            "MCHC",
            "Platelet_Count",
            "RDW",
            "MCH",
            "Basophils",
            "Eosinophils",
            "Lymphocytes",
            "Monocytes",
            "Neutrophils",
        ]

        return df.drop(columns=drop_cols, errors="ignore")

    # %%
    # ---- make full_dict
    full_dict = {}
    for k, v in cases_features_dict.items():
        full_dict[k] = pd.concat(
            [cases_features_dict[k], controls_features_dict[k]], axis=0
        )
        if SAVE_CASE_CONTROL_STAT:
            save_stat(full_dict[k], k, MATCH_DATE)

    for k, v in full_dict.items():
        full_dict[k] = full_dict[k].drop(columns=ignore_cols)
        # full_dict[k] = drop_features(full_dict[k])

    for k, v in full_dict.items():
        nan_imputation(v, LAB_ITEMS=LAB_ITEMS, USE_LAB=USE_LAB)

    if USE_LAB:
        for col in LAB_ITEMS.values():
            for k, v in full_dict.items():  # Removing 결측 플래그
                full_dict[k] = full_dict[k].drop(columns=[f"{col}_miss"])

    # %%
    if verbose:
        print(full_dict["E"]["label"])

        print(f"full_dict의 컬럼 (USE_LAB={USE_LAB}): {full_dict['G'].columns}")
        print(f"full_dict의 컬럼 수: {len(full_dict['G'].columns)}")

    def make_train_and_test(
        full, g_seed, test_frac
    ):  # full 을 받아서 train_df(85%), test_df(15%)로 스플릿
        """
        Stratified train/test split

        Args:
            full (pd.DataFrame): 전체 데이터 (label 컬럼 포함)
            g_seed (int): random seed

        Returns:
            train_df (pd.DataFrame), test_df (pd.DataFrame)
        """

        train_df, test_df = train_test_split(
            full,
            test_size=test_frac,
            stratify=full["label"],  # label 분포 유지 /full의 'label'컬럼만 넣음
            random_state=g_seed,
            shuffle=True,
        )

        return train_df, test_df

    def make_cv_splits_from_train_valid(
        full_df: pd.DataFrame, n_splits: int = 5, random_state: int = split_seed
    ):

        X = np.arange(len(full_df))
        y = full_df["label"].values

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        folds = []
        n_iter = 0
        for tr_idx, va_idx in skf.split(X, y):
            n_iter += 1
            label_tr = full_df["label"].iloc[tr_idx]
            label_va = full_df["label"].iloc[va_idx]
            print(f"## 교차 검증: {n_iter}")
            print("학습 레이블 데이터 분포:\n", label_tr.value_counts())
            print("검증 레이블 데이터 분포:\n", label_va.value_counts())

            train_cv = full_df.iloc[tr_idx].copy()
            valid_cv = full_df.iloc[va_idx].copy()

            # ----- 스케일링: 매 fold의 train에 맞춰 학습 -----
            folds.append((n_iter, train_cv, valid_cv))
        return folds

    train_df_dict = {}
    valid_df_dict = {}
    test_df_dict = {}
    cv_folds_dict = {}
    for k, v in full_dict.items():
        train_df_dict[k], test_df_dict[k] = make_train_and_test(
            full_dict[k], g_seed if FIX_TEST_SPLIT else split_seed, test_frac=0.15
        )
        scaler = normalize(train_df_dict[k], [test_df_dict[k]])  #
        if SAVE_SCALER:
            save_scaler(scaler, f"scaler_{k}.json")

        # Make valid set from train set
        if meta_config["use_cv"] is False:
            train_df_dict[k], valid_df_dict[k] = make_train_and_test(
                train_df_dict[k], split_seed, test_frac=0.2
            )
        else:
            cv_folds_dict[k] = make_cv_splits_from_train_valid(
                train_df_dict[k],
                n_splits=meta_config["N_SPLITS"],
                random_state=split_seed,
            )
    return train_df_dict, valid_df_dict, test_df_dict, cv_folds_dict, full_dict
