import numpy as np
import pandas as pd


def cast_basic_type(value):
    """
    value를 bool, int, float, str 중 하나의 Python 기본 타입으로 변환한다.
    numpy 타입도 처리 가능.
    """
    # numpy -> python 기본 타입 변환
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    elif isinstance(value, (np.integer, int)):
        return int(value)
    elif isinstance(value, (np.floating, float)):
        return float(value)
    elif isinstance(value, (str, np.str_)):
        return str(value)
    else:
        # 기본 타입이 아니면 문자열로 강제 변환
        return str(value)


def drop_constant_columns(df: pd.DataFrame):

    removed_cols = {}
    drop_list = []

    for col in df.columns:
        if df[col].nunique(dropna=False) == 1:  # 모든 값이 동일
            removed_cols[col] = df[col].iloc[0]  # 컬럼 이름: 값 쌍으로 저장
            drop_list.append(col)

    new_df = df.drop(columns=drop_list)
    removed_cols_converted = {k: cast_basic_type(v) for k, v in removed_cols.items()}
    return new_df, removed_cols_converted
