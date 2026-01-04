import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from predcancer.manage_json import SORT_KEYS, modify_dict
from predcancer.utils import (check_duplicate_in_dict_of_lists,
                              get_sens_spec_f1, my_p_grid,
                              remove_duplicates_in_list_of_dicts, save_np)

SEED_FIRST = True
SAVE_CASE_CONTROL_STAT = False
FIX_TEST_SPLIT = True  # if True, use fixed test split for all experiments
SAVE_SCALER = True
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42

SAVE_DATA_SPLIT_DF_FOR_MODEL_INPUT = False
# If True, save train/test split dataframes for model input to `saved_dataloaders/`


class MK:
    DEBUG = "debug"
    DEFAULT = "default"
    LARGE = "large"  # large hyperparameter space
    RANDOM_SPLIT = "random_split"
    RANDOM_SPLIT_VARY_FT_LR = "random_split_vary_ft_lr"
    RANDOM_SPLIT_VARY_FT_LR2 = "random_split_vary_ft_lr2"
    RANDOM_SPLIT_VARY_FT_LR3 = "random_split_vary_ft_lr3"
    RANDOM_SPLIT_VARY_FT_LR4 = "random_split_vary_ft_lr4"
    RANDOM_SPLIT_VARY_FT_LR5 = "random_split_vary_ft_lr5"
    RANDOM_SPLIT_VARY_FT_LR6 = "random_split_vary_ft_lr6"
    FASTER = "faster"


MODE = MK.RANDOM_SPLIT_VARY_FT_LR6


class ResK:  # metric keys
    AUC = "auc"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    F1 = "f1"
    BRIER = "brier"
    THRES = "thres"
    THRES_F1 = "thres_f1"
    PROB = "prob"


DEBUG = MODE == MK.DEBUG
DEBUG2 = False


def get_out_dir():
    if MODE == MK.DEBUG or DEBUG2:
        return Path(f"debug_results_0")
    else:
        return Path("results")


out_dir = get_out_dir()
DEFAULT_FINAL_DIR = "final_result"
DEFAULT_PRE_DIR = "pretrain"
BASE_EXP_OUT_DIR = "exp_out"

PROJECT_DIR = f"."
STOP_FILE_PATH = os.path.join(PROJECT_DIR, "stop.txt")
MERGED_DIR = os.path.join(PROJECT_DIR, "merged")

CONTROL_RATIO = 3  # 1:3 case:control as recommended
g_seed = 42
USE_CV = False

SPLIT_SEEDS = list(range(0, 10))


def get_mode_dict():
    ofat = {}
    best = {}
    data = {
        MK.DEBUG: {
            "seed": [0],
            "ft_epoch": [50],
            "ft_lr": [1e-4],
            "reg_mode": ["weight"],
            "ft_weight_decay": [1e-2],
            "sp_reg_coeff": [1e-2],
            "sp_reg": [True],
        },
        MK.DEFAULT: {
            "ft_head_only": [True, False],
            "lora": [False],
            "seed": list(range(10)),
            "sp_reg": [True, False],
            "ft_epoch": [50, 100, 200],
            "ft_lr": [1e-4],
            "ft_weight_decay": [1.0, 1e-1, 1e-2],
            "sp_reg_coeff": [0.1, 1e-2, 1e-3, 1e-4, 1e-5],
        },
    }
    best[MK.DEFAULT] = {
        "ft_weight_decay": [1.0],
        "ft_epoch": [100],
        "ft_lr": [1e-4],
        "sp_reg": [True],
        "ft_head_only": [False],
    }
    data[MK.LARGE] = {
        **deepcopy(data[MK.DEFAULT]),
        **{
            "ft_epoch": [50, 100, 200, 500],
            "ft_lr": [1e-3, 1e-4, 1e-5],
        },
    }
    best[MK.LARGE] = deepcopy(best[MK.DEFAULT])

    data[MK.RANDOM_SPLIT] = {
        **deepcopy(data[MK.DEFAULT]),
        **{
            "seed": [0],
            "ft_epoch": [50, 100, 200],
            "split_seed": SPLIT_SEEDS,
        },
    }
    best[MK.RANDOM_SPLIT] = deepcopy(best[MK.DEFAULT])

    data[MK.RANDOM_SPLIT_VARY_FT_LR] = {
        **deepcopy(data[MK.RANDOM_SPLIT]),
        **{
            # "ft_epoch": [50, 100, 200],
            "ft_lr": [1e-3, 1e-4, 1e-5],
        },
    }
    best[MK.RANDOM_SPLIT_VARY_FT_LR] = deepcopy(best[MK.RANDOM_SPLIT])
    data[MK.RANDOM_SPLIT_VARY_FT_LR2] = {
        **deepcopy(data[MK.RANDOM_SPLIT_VARY_FT_LR]),
        **{
            "ft_weight_decay": [10.0, 1.0, 1e-1],
            "sp_reg_coeff": [1e-3, 1e-4, 1e-5, 1e-6],
        },
    }
    best[MK.RANDOM_SPLIT_VARY_FT_LR2] = deepcopy(best[MK.RANDOM_SPLIT_VARY_FT_LR])
    data[MK.RANDOM_SPLIT_VARY_FT_LR3] = {
        **deepcopy(data[MK.RANDOM_SPLIT_VARY_FT_LR2]),
        **{
            "ft_weight_decay": [10.0, 1.0, 1e-1],
            "sp_reg_coeff": [1e-4, 1e-5, 1e-6],
            "ft_lr": [1e-4, 1e-5, 1e-6],
            "ft_epoch": [100, 200, 500],
        },
    }
    best[MK.RANDOM_SPLIT_VARY_FT_LR3] = deepcopy(best[MK.RANDOM_SPLIT_VARY_FT_LR2])
    data[MK.RANDOM_SPLIT_VARY_FT_LR4] = {
        **deepcopy(data[MK.RANDOM_SPLIT_VARY_FT_LR3]),
        **{
            "ft_weight_decay": [10.0, 1.0, 1e-1],
            "sp_reg_coeff": [1e-4, 1e-5, 1e-6],
            "ft_lr": [1e-4, 1e-5, 1e-6],
            "ft_epoch": [50, 100, 200],
            "reg_mode": ["weight", "all"],
        },
    }
    best[MK.RANDOM_SPLIT_VARY_FT_LR4] = {
        "ft_weight_decay": [0.1],
        "ft_epoch": [200],
        "ft_lr": [1e-4],
        "sp_reg": [False],
        "ft_head_only": [False],
    }
    data[MK.RANDOM_SPLIT_VARY_FT_LR5] = {
        **deepcopy(data[MK.RANDOM_SPLIT_VARY_FT_LR4]),
        **{
            "sp_reg": [False],
            "ft_weight_decay": [1e-4],
            "ft_lr": [1e-4],
            "ft_epoch": [50],
            "reg_mode": ["weight"],
        },
    }
    best[MK.RANDOM_SPLIT_VARY_FT_LR5] = deepcopy(best[MK.RANDOM_SPLIT_VARY_FT_LR4])
    curr_k = MK.RANDOM_SPLIT_VARY_FT_LR6
    data[curr_k] = {
        **deepcopy(data[MK.RANDOM_SPLIT_VARY_FT_LR5]),
        **{
            "reg_mode": ["all"],
        },
    }
    best[curr_k] = deepcopy(best[MK.RANDOM_SPLIT_VARY_FT_LR5])
    ofat[curr_k] = {
        "layer": [[128, 64], [64, 32], [32, 16]],
        "max_steps": [1000, 2000, 5000],
        "pre_weight_decay": [1e-2, 1e-3, 1e-4],
        "pre_lr": [1e-2, 1e-3, 1e-4],
        "ft_weight_decay": [1e-3, 1e-4, 1e-5],
        "ft_lr": [1e-3, 1e-4, 1e-5],
        "ft_epoch": [20, 50, 100],
    }
    for k, v in ofat[curr_k].items():
        if len(v) == 3:
            best[curr_k][k] = [v[1]]

    curr_k = MK.FASTER
    data[curr_k] = {
        **deepcopy(data[MK.RANDOM_SPLIT_VARY_FT_LR6]),
        **{
            "layer": [[128, 64]],
            "max_steps": [2000],
            "pre_weight_decay": [1e-3],
            "pre_lr": [1e-3],
            "ft_weight_decay": [1e-4],
            "ft_lr": [1e-4],
            "ft_epoch": [200],
            "split_seed": [0, 1, 2],
            "ft_head_only": [False],
            "lora": [True, False],
        },
    }
    best[curr_k] = {}
    for k, v in data[curr_k].items():
        if len(v) == 1:
            best[curr_k][k] = v
    out = data[MODE]
    if USE_CV:
        assert len(out["seed"]) == 1  # only support one seed for CV now
    if DEBUG2:
        out = deepcopy(out)
        out["seed"] = out["seed"][:2]
        out["split_seed"] = out["split_seed"][:2]
        for k, v in out.items():
            if len(v) >= 3:
                out[k] = v[1:2]
    return out, best.get(MODE), ofat.get(MODE)


MODE_DICT, BEST, OFAT = get_mode_dict()


def get_split_seeds(): ...


BATCH_SIZE = 64
STEPS_PER_EPOCH = 100  # 5325/64

USE_EGD = False
USE_LAB = True
USE_ED = False

num_cols = ["AGE"]

TIME_SPLIT = False  # 이 플래그로 시간 분할을 제어함
MATCH_DATE = False
DAYS_GAP = 30
# DATA_VERSION = "2025-09-09-v2"
# DATA_VERSION = "2025-10-09"
# DATA_VERSION = "2025-10-09-v2"
DATA_VERSION = (
    "2025-10-09-v4"  # default, 이 버전 기준으로 동작하지 않는 부분 preprocess에서 삭제
)

# DATA_VERSION = "2025-10-09-v5"
# DATA_VERSION = "2025-10-09-v6"

# DATA_VERSION = "2025-10-09-v7"
# lab-complete filtering is disabled from `"2025-10-09-v4"`

ignore_cols = ["admittime_GAP"] + ["subject_id", "c_time"]

RUN_SHAP = False

GLOBAL_OUTER_HP_SPACE = {}
GLOBAL_INNER_HP_SPACE = {
    "seed": MODE_DICT["seed"],
}
_SMALL_XGB_SPACE = {
    "learning_rate": [0.3],  # log-uniform 샘플 권장  # (0.02, 0.2)
    "max_depth": [6],  # [3, 4, 5, 6]
    "min_child_weight": [1],  # [1, 2, 4, 6, 8]
    "subsample": [1.0],  # (0.6, 1.0)
    "colsample_bytree": [1.0],  # 0.6, 1.0
    "reg_lambda": [1e-1, 1, 10],
    "gamma": [0.0],  # min_split_loss # 0.0, 5.0
    "scale_pos_weight": [float(CONTROL_RATIO)],
}
_LARGE_XGB_SPACE = {
    "learning_rate": [0.01, 0.1, 1.0],  # log-uniform 샘플 권장  # (0.02, 0.2)
    "max_depth": [2, 4, 6],  # [3, 4, 5, 6]
    "min_child_weight": [1, 3, 5],  # [1, 2, 4, 6, 8]
    "subsample": [0.6, 1.0],
    "colsample_bytree": [0.6, 1.0],
    "reg_lambda": [1e-2, 1e-1, 1],
    "gamma": [0.0, 5.0],
    "scale_pos_weight": [float(CONTROL_RATIO)],
}


PARAM_SPACES = {
    "MLP": {
        "model": ["MLP"],
        "pos_weight": [float(CONTROL_RATIO)],
        "lr": [1e-1, 1e-2, 1e-3],
        "weight_decay": [1e-3, 1e-4, 1e-5],
        # "lr": [1e-3],
        # "weight_decay": [1e-3],
        "layer": [[32, 16], [16, 8], [8, 4]],
        "epoch": [10, 20, 50],
        "dropout": [0.2],
        "use_es": [False],
        "mb_rate": [0.05],  # None
        "drop_last": [True],
    },
    "LR": {
        "model": ["LR"],
        "c": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "class_weight": [None, "balanced"],
    },
    "XGB": {"model": ["XGB"], **_LARGE_XGB_SPACE},
    "Transfer": {
        "model": ["Transfer"],
        "valid_steps": [STEPS_PER_EPOCH],
        "pos_weight": [float(CONTROL_RATIO)],
        "layer": MODE_DICT.get("layer", [[64, 32]]),
        "dropout": [0.2],
        "cancer_type": [
            ["C", "E", "L", "P"],
            # ["C"],
            # ["E"],
            # ["L"],
            # ["P"],
        ],
        "max_steps": [2000],
        "use_epoch_loader": [False],
        "pre_lr": [1e-3],  # [1e-2, 1e-3],
        "pre_weight_decay": [1e-3],
        "pre_use_es": [False],
        "ft_epoch": MODE_DICT["ft_epoch"],
        "ft_lr": MODE_DICT["ft_lr"],
        "ft_weight_decay": MODE_DICT["ft_weight_decay"],
        "ft_use_es": [False],
        "head_mode": ["new"],  # best, new, avg
        "ft_head_only": MODE_DICT["ft_head_only"] if not DEBUG else [False],
        "sp_reg": MODE_DICT["sp_reg"],
        "mb_rate": [0.05],  # None
        "reg_mode": (
            ["all"] if "reg_mode" not in MODE_DICT else MODE_DICT["reg_mode"]
        ),  # "weight", "all"
        "drop_last": [True],
        "lora": MODE_DICT["lora"],
    },
}


def get_models():
    if DEBUG or MODE == MK.FASTER:
        return ["Transfer"]
    else:
        return ["Transfer", "LR", "MLP", "XGB"]


MODELS = get_models()


def get_model_h_p_space_list(
    do_transfer_ablation=not (MODE == MK.DEBUG or DEBUG2), vary_sp_reg=False
) -> List[Dict[str, list]]:
    gc_rate_space = {
        "gc_rate": (
            [1.0] if DEBUG or DEBUG2 or MODE in [MK.FASTER] else [1.0, 0.5, 0.2, 0.1]
        )  # , 0.05, 0.02, 0.01]
    }
    output = [{**gc_rate_space, **PARAM_SPACES[m]} for m in MODELS]
    if "Transfer" in MODELS and do_transfer_ablation:
        default = deepcopy(PARAM_SPACES["Transfer"])

        # Sweep pretraining hyperparameters
        if OFAT is not None:
            for k, v in OFAT.items():
                output.append({**deepcopy(default), **gc_rate_space, k: v})

        # "sp_reg": [True, False],
        if vary_sp_reg:
            output.append(
                {**deepcopy(default), **gc_rate_space, "sp_reg": [False, True]}
            )

        # "ft_head_only": [True, False],
        output.append(
            {**deepcopy(default), **gc_rate_space, "ft_head_only": [True, False]}
        )

        # "cancer_type"
        if MODE not in [MK.FASTER]:
            output.append(
                {
                    **deepcopy(default),
                    **({} if BEST is None else BEST),
                    "gc_rate": [1.0],
                    "cancer_type": [
                        ["C", "E", "L", "P"],
                        ["C"],
                        ["E"],
                        ["L"],
                        ["P"],
                        # ["C", "E"],
                        # ["C", "L"],
                        # ["C", "P"],
                        # ["E", "L"],
                        # ["E", "P"],
                        # ["L", "P"],
                        # ["C", "E", "L"],
                        # ["C", "E", "P"],
                        # ["C", "L", "P"],
                        # ["E", "L", "P"],
                    ],
                }
            )

    return output


MODEL_H_P_SPACE_LIST = get_model_h_p_space_list()


def get_meta_config():
    output = {
        "data_version": DATA_VERSION,
        "use_lab": USE_LAB,
        "match_date": MATCH_DATE,
        "days_gap": DAYS_GAP,
        "control_ratio": CONTROL_RATIO,
        "use_cv": USE_CV,
        # "N_SPLITS": 5,  # 원하는 fold 수
        # 제거할 열 리스트
        "drop_cols": [
            "NEOPLASM_INTRA_ABDOMINAL_LYMPH",
            "NEOPLASM_RETROPERITONEUM_PERITONEUM",
        ],
    }
    if FIX_TEST_SPLIT:
        output["fix_test_split"] = True
    return output


META_CONFIG = get_meta_config()


def get_param_space(model_name, use_cv=USE_CV):
    output = PARAM_SPACES[model_name]
    if use_cv:
        output["cv_id"] = list(range(1, META_CONFIG["N_SPLITS"] + 1))
        # Fold indices for cross-validation
    return {**GLOBAL_OUTER_HP_SPACE, **output, **GLOBAL_INNER_HP_SPACE}


def check_hyper_setting(hyper_setting, meta_config):
    if len(hyper_setting["seed"]) >= 2:
        assert "N_SPLITS" not in meta_config
        assert "cv_id" not in hyper_setting
    if "cv_id" in hyper_setting:
        assert "N_SPLITS" in meta_config
        n_splits = meta_config["N_SPLITS"]
        assert len(hyper_setting["seed"]) == 1
        assert isinstance(n_splits, int)
        assert hyper_setting["cv_id"] == list(
            range(1, n_splits + 1)
        ), f"setting cv_id {hyper_setting['cv_id']} is invalid for N_SPLITS={list(range(1, n_splits + 1))}"


meta_config = {**META_CONFIG}


def get_param_space_list(use_cv=USE_CV):
    out = []
    for p in MODEL_H_P_SPACE_LIST:
        if use_cv:
            p = {**p, "cv_id": list(range(1, META_CONFIG["N_SPLITS"] + 1))}
        hp_space = {**GLOBAL_OUTER_HP_SPACE, **p, **GLOBAL_INNER_HP_SPACE}
        check_hyper_setting(hp_space, meta_config)
        check_duplicate_in_dict_of_lists(hp_space)
        out.append(hp_space)
    return out


HP_SPACE_LIST = get_param_space_list()


def get_batch_size(data_size, mb_rate, max_bs=64, min_bs=8):
    bs = int(data_size * mb_rate)
    bs = min(bs, max_bs)
    bs = max(bs, min_bs)
    print(f"Using batch size {bs} for data size {data_size} and mb_rate {mb_rate}")
    return bs


def get_file_name(num):
    filename = f"{num}.json"
    return filename


def get_file_path(out_dir, num):
    return out_dir / get_file_name(num)


# ---- graceful stop: if we catch the stop signal from the `STOP_FILE_PATH` to finish current run then halt ----
def stop_requested() -> bool:
    p = Path(STOP_FILE_PATH)
    if p.exists():
        with open(p, "r") as f:
            val = f.read().strip()
        if val == "1":
            print(
                f"STOP marker (in `stop.txt`) detected before next run. Halting after previous run."
            )
            return True
    return False


def diverge_settings(setting: dict) -> List[dict]:
    setting = deepcopy(setting)
    if setting.get("ft_head_only") is True and (setting["sp_reg"]):
        return []
    if setting.get("sp_reg") is True:
        out = []
        for sp_reg_coeff in MODE_DICT["sp_reg_coeff"]:
            out.append({**setting, "sp_reg_coeff": sp_reg_coeff})
    else:
        out = [setting]
    return out


def safe_my_param_grid_with_modification(param_grid, meta_config):
    out = []
    for setting in my_p_grid(param_grid, meta_config):
        check_sanity(setting)
        modify_dict(setting, SORT_KEYS)
        s_list = diverge_settings(setting)
        out += s_list
    if "split_seed" in MODE_DICT:
        _out = []
        if SEED_FIRST:
            for split_seed in MODE_DICT["split_seed"]:
                for s in out:
                    _out.append({**s, "split_seed": split_seed})
        else:
            for s in out:
                for split_seed in MODE_DICT["split_seed"]:
                    _out.append({**s, "split_seed": split_seed})
        out = _out
    out = remove_duplicates_in_list_of_dicts(out)
    return out


def get_final_transfer_model_path(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "Transfer_best_model.pt")


def get_exp_out_dir(exp_name: str) -> str:
    out_dir = os.path.join(BASE_EXP_OUT_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_infer_prob_path(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "probs.csv")


def get_infer_prob_np_path(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "probs.npy")


def get_label_np_path(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "labels.npy")


def get_val_thres_np_path(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "val_thres.npy")


def get_val_thres_f1_np_path(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "val_thres_f1.npy")


def check_sanity(d: dict):
    """Check that d has expected keys and types."""
    assert "use_cv" in d and isinstance(d["use_cv"], bool)
    if d["use_cv"] is False:
        assert "cv_id" not in d
        assert "N_SPLITS" not in d
    else:
        assert "cv_id" in d
        assert "N_SPLITS" in d and isinstance(d["N_SPLITS"], int) and d["N_SPLITS"] > 1


def get_pretrained_model_path(dir_path: str | Path, filename) -> Path:
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    return dir_path / "data" / f"{filename}.pt"


def get_trained_model_path(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "model.pt")


DF_CSV_DIR = "saved_dataloaders"
TRAIN_DF_CSV_PATH = os.path.join(DF_CSV_DIR, "train_df.csv")
TEST_DF_CSV_PATH = os.path.join(DF_CSV_DIR, "test_df.csv")


def get_train_df_csv_path():
    os.makedirs(DF_CSV_DIR, exist_ok=True)
    return TRAIN_DF_CSV_PATH


def get_test_df_csv_path():
    os.makedirs(DF_CSV_DIR, exist_ok=True)
    return TEST_DF_CSV_PATH


def from_test_probs(
    exp_name: str,
    y_pred_prob_test: np.ndarray,
    y_test: np.ndarray,
    val_thres,
    val_thres_f1,
):
    assert isinstance(y_pred_prob_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    test_auc = roc_auc_score(y_test, y_pred_prob_test)
    test_ap = average_precision_score(y_test, y_pred_prob_test)
    test_brier = brier_score_loss(y_test, y_pred_prob_test)
    test_sensitivity, test_specificity, test_f1, _, _ = get_sens_spec_f1(
        y_test, y_pred_prob_test, val_thres, val_thres_f1
    )
    exp_out_dir = get_exp_out_dir(exp_name)
    y_pred_path = get_infer_prob_np_path(exp_out_dir)
    y_true_path = get_label_np_path(exp_out_dir)
    val_thres_path = get_val_thres_np_path(exp_out_dir)
    val_thres_f1_path = get_val_thres_f1_np_path(exp_out_dir)
    if not os.path.exists(y_pred_path):
        save_np(y_pred_prob_test, y_pred_path)
    if not os.path.exists(y_true_path):
        save_np(y_test, y_true_path)
    if not os.path.exists(val_thres_path):
        save_np(np.array([val_thres], dtype=float), val_thres_path)
    if not os.path.exists(val_thres_f1_path):
        save_np(np.array([val_thres_f1], dtype=float), val_thres_f1_path)
    return {
        "test_auroc": test_auc,
        "test_ap": test_ap,
        "test_sensitivity": test_sensitivity,
        "test_specificity": test_specificity,
        "test_f1": test_f1,
        "test_brier": test_brier,
    }
