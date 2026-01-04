import pandas as pd


def count_nan(df):
    for c in df.columns:
        print(c, df[c].isna().sum())


def nan_imputation(df: pd.DataFrame, LAB_ITEMS, USE_LAB):

    num_cols = ["AGE"]
    if USE_LAB:
        num_cols += list(LAB_ITEMS.values())  # numeric cols list

    if USE_LAB:
        for col in LAB_ITEMS.values():
            # f"{col}_miss"라는 결측 플래그 열 생성, 결측치:1/아님:0
            df[f"{col}_miss"] = df[col].isna().astype("int8")

    meds = df.loc[df.label == 0, num_cols].median()
    # 컨트롤 기준 중앙값 (meds는 pd.Series)
    df[num_cols] = df[num_cols].fillna(meds)

    # for col in LAB_ITEMS.values():  # Removing 결측 플래그 columns
    #     df = df.drop(columns=[f"{col}_miss"])

    return df
    # num_cols_df = full[[col for col in LAB_ITEMS.values()]]  # Extracting lab columns

    # for col in LAB_ITEMS.values():          # Removing lab columns
    # full = full.drop(columns=[f"{col}"])
