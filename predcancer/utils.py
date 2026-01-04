import json
import random
import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping

import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

# def dic_match(d1: dict, d2: dict):
#     """
#     Compare two dictionaries to check if all key–value pairs in `d2`
#     are present and equal in `d1`.
#     """


#     for k, v in d2.items():
#         if d1[k] != d2[k]:
#             return False
#     return True
def compare_dicts(dict1: dict, dict2: dict):
    """
    Compare two flat (non-nested) dictionaries and return differences.

    Returns:
        differences: dict
            {
                "only_in_dict1": [...],
                "only_in_dict2": [...],
                "different_values": {key: (value_in_dict1, value_in_dict2)}
            }
    """
    # dict1에는 있고 dict2에는 없는 key
    only_in_dict1 = set(dict1.keys()) - set(dict2.keys())
    # dict2에는 있고 dict1에는 없는 key
    only_in_dict2 = set(dict2.keys()) - set(dict1.keys())
    # key는 같지만 value가 다른 경우
    different_values = {
        key: (dict1[key], dict2[key])
        for key in dict1.keys() & dict2.keys()
        if dict1[key] != dict2[key]
    }

    return {
        "only_in_dict1": list(only_in_dict1),
        "only_in_dict2": list(only_in_dict2),
        "different_values": different_values,
    }


def drop_results(result):
    keys_to_drop = get_result_cols(result)
    for key in keys_to_drop:
        if key in result:
            del result[key]
    return result


def extract_config(dic: dict):
    out = deepcopy(dic)
    drop_results(out)
    keys_to_drop = ["exp_name", "filename"]
    for key in keys_to_drop:
        if key in out:
            del out[key]
    return out


def file_path_to_exp_name(file_path: str) -> str:
    # 확장자를 제외하고 문자열 끝의 숫자 추출
    stem = Path(file_path).stem
    number_str = re.search(r"\d+$", stem).group()
    return number_str


def get_device(RANK=None):
    if torch.cuda.is_available() and (RANK is not None):
        torch.cuda.set_device(0)  # 마스킹된 환경에선 항상 0
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_result_cols(col_names):
    split_names = ["val", "test"]
    split_metric_names = []
    for s in split_names:
        for m in [
            "auroc",
            "ap",
            "sensitivity",
            "specificity",
            "f1",
            "accuracy",
            "brier",
        ]:
            split_metric_names.append(f"{s}_{m}")
    final_metric_names = split_metric_names + [
        "best_epoch",
        "best_step",
        "selected_type",
        "thres",
        "ft_elapsed_sec",
    ]
    out = []
    # Append columns that contain any of the final metric names
    for col_name in col_names:
        if any(metric_name in col_name for metric_name in final_metric_names):
            out.append(col_name)
    return out


def get_metric_cols(col_names):
    out = get_result_cols(col_names)
    futher_exclude_cols = ["selected_type"]
    out2 = []
    for out_col_name in out:
        if any(
            exclude_col_name not in out_col_name
            for exclude_col_name in futher_exclude_cols
        ):
            out2.append(out_col_name)
    return out2


def dic_match(d1: dict, d2: dict):
    for k, v in d2.items():
        if k not in d1 or d1[k] != v:
            return False
    return True


def seed_init(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # print(f"Seed initialized to {seed}")


class GCDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(
            df.drop(columns=["label"]).values,  ### df 칼럼 list 확인
            dtype=torch.float32,
        )
        self.y = torch.tensor(df.label.values.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class InfiniteDataLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)  # epoch 끝나면 새 iterator
            return next(self.iterator)


def my_tqdm(*args, **kwargs):
    for x in tqdm(
        *args,
        **kwargs,
        smoothing=0.3,
        dynamic_ncols=True,
    ):
        yield x


def get_best_f1_threshold(y_true, y_pred_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)

    # thresholds[i] ↔ (precisions[i+1], recalls[i+1])
    p = precisions[1:]
    r = recalls[1:]

    if len(thresholds) == 0 or len(p) == 0:
        # 모든 샘플이 한 클래스인 극단 케이스 등
        return 0.5, 0.0

    f1_scores = 2 * (p * r) / (p + r + 1e-12)

    best_idx = np.argmax(f1_scores)
    best_threshold: float = thresholds[best_idx].item()
    best_f1_curve: float = f1_scores[best_idx].item()  # 참고값(커브 상 F1)

    return best_threshold, best_f1_curve


def threshold_closest_to_perfect(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)  # all three have same length
    d2 = (1 - tpr) ** 2 + (fpr) ** 2  # squared distance
    i = int(np.argmin(d2))
    return {
        "threshold": float(thr[i]),
        "sensitivity": float(tpr[i]),
        "specificity": float(1 - fpr[i]),
        "fpr": float(fpr[i]),
        "distance": float(np.sqrt(d2[i])),
        "index": i,
    }


def get_sens_spec_f1(y_true, y_pred_prob, threshold=None, thres_f1=None):
    """
    threshold가 None이면 F1을 최대화하는 threshold를 찾아 사용.
    반환: sensitivity(=recall), specificity, f1(실제 예측기준)
    """
    if threshold is None:
        threshold = threshold_closest_to_perfect(y_true, y_pred_prob)["threshold"]
    if thres_f1 is None:
        thres_f1, _ = get_best_f1_threshold(y_true, y_pred_prob)

    y_pred = (y_pred_prob >= threshold).astype(int)

    # 항상 2x2 행렬을 받도록 labels 고정
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-12)  # recall
    specificity = tn / (tn + fp + 1e-12)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return sensitivity, specificity, f1, threshold, thres_f1


def deep_freeze(obj: Any) -> Hashable:
    """
    Convert a possibly nested Python object (dict/list/tuple/set) into
    an immutable, hashable, *order-invariant for dicts/sets* representation.

    Rules:
    - dict -> tuple of (key, frozen(value)) sorted by key
    - list/tuple -> tuple of frozen elements (order-preserving)
    - set -> tuple of frozen elements sorted to be order-invariant
    - other -> returned as-is (must already be hashable)

    Note:
    - This matches Python's dict equality semantics (key order ignored).
    - Leaves must be hashable (e.g., str, int, float, bool, None).
      If you have unhashable leaves (e.g., numpy arrays), convert them
      to hashable forms beforehand or extend this function accordingly.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, deep_freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(deep_freeze(x) for x in obj)
    if isinstance(obj, set):
        return tuple(sorted(deep_freeze(x) for x in obj))
    return obj  # assumed hashable


def check_duplicate_in_dict_of_lists(d: dict):
    """
    Check for duplicates in the lists of a dictionary.
    Args:
        d (dict): A dictionary where each value is a list.
    Returns:
        dict: A dictionary with keys as the original keys and values as lists of duplicates found.
    """
    for key, lst in d.items():
        seen = set()
        for item in lst:
            item = deep_freeze(item)
            if item in seen:
                raise Exception(
                    f"Duplicate item '{item}' found in list for key '{key}'"
                )


def remove_duplicates_in_list_of_dicts(dicts: list[dict]) -> list[dict]:
    """
    Given a list of dictionaries, return a new list with duplicates removed.
    Two dictionaries are considered duplicates if their contents are identical
    after normalization (i.e., after removing ignored keys and sorting lists).
    The first occurrence is kept.
    """
    seen = set()
    unique_dicts = []

    for d in dicts:
        h = deep_freeze(extract_config(d))
        if h not in seen:
            seen.add(h)
            unique_dicts.append(d)

    return unique_dicts


def my_p_grid(param_grid, meta_config=None):
    """
    Almost identical to `sklearn.model_selection.ParameterGrid`,
    except that the order of keys is preserved.
    """
    if meta_config is None:
        meta_config = {}
    if not isinstance(param_grid, (Mapping, Iterable)):
        raise TypeError(
            f"Parameter grid should be a dict or a list, got: {param_grid!r} of"
            f" type {type(param_grid).__name__}"
        )

    if isinstance(param_grid, Mapping):
        # wrap dictionary in a singleton list to support either dict
        # or list of dicts
        param_grid = [param_grid]

    for p in param_grid:
        # Always sort the keys of a dictionary, for reproducibility
        items = p.items()
        if not items:
            yield dict(meta_config)
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield {**meta_config, **params}


def save_scaler(scaler: StandardScaler, filepath: str | Path) -> None:
    """
    Save a fitted StandardScaler to a JSON file (version-independent).

    Parameters
    ----------
    scaler : StandardScaler
        A fitted sklearn StandardScaler object.
    filepath : str or Path
        Output path to save the scaler parameters as JSON.
    """
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("The provided scaler is not fitted yet.")

    params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"✅ Scaler saved to {filepath}")


def load_scaler(filepath: str | Path) -> StandardScaler:
    """
    Load a StandardScaler from a JSON file (version-independent).

    Parameters
    ----------
    filepath : str or Path
        Path to the JSON file containing scaler parameters.

    Returns
    -------
    StandardScaler
        A restored StandardScaler object ready for transform().
    """
    with open(filepath, "r", encoding="utf-8") as f:
        params: dict = json.load(f)

    scaler = StandardScaler(
        with_mean=params.get("with_mean", True), with_std=params.get("with_std", True)
    )

    scaler.mean_ = np.array(params["mean"], dtype=float)
    scaler.scale_ = np.array(params["scale"], dtype=float)
    scaler.var_ = np.array(params["var"], dtype=float)
    scaler.n_features_in_ = len(scaler.mean_)  # ← 자동 복원

    return scaler


def save_np(a: np.ndarray, path: str):
    assert isinstance(a, np.ndarray)
    with open(path, "wb") as f:
        np.save(f, a)


def load_np(path):
    with open(path, "rb") as f:
        a = np.load(f)
    return a
