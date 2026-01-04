# %%
# -------------- 0.  Imports & constants -----------------
import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from predcancer.manage_json import ConfigManager
from predcancer.merge_results import make_merged_results
from predcancer.preprocess import run_preprocess
from predcancer.run_single_experiment import run_single_experiment
from predcancer.settings import (HP_SPACE_LIST, STOP_FILE_PATH, g_seed,
                                 get_file_name, get_file_path, meta_config,
                                 out_dir, safe_my_param_grid_with_modification,
                                 stop_requested)
from predcancer.utils import extract_config, get_device, my_tqdm, seed_init

tqdm.pandas()

import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--rank", type=int, default=0, help="process index / GPU index (0-based)"
)
_parser.add_argument(
    "--world_size", type=int, default=1, help="total number of parallel processes/GPUs"
)
_parser.add_argument("--pretrain", action="store_true")  # on/off flag
try:
    _args, _ = _parser.parse_known_args()
    RANK = _args.rank
    WORLD_SIZE = _args.world_size
except SystemExit:
    RANK, WORLD_SIZE = 0, 1
DO_PRETRAIN = _args.pretrain
print(f"[Parallel] RANK={RANK} WORLD_SIZE={WORLD_SIZE}")


seed_init(g_seed)
# if torch.cuda.is_available() and (RANK is not None):
#     torch.cuda.set_device(RANK)  # RANK 번째 GPU 사용
#     device = torch.device(f"cuda:{RANK}")
# else:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device(RANK)

# LEARN_BEST_SETTING = True
# if LEARN_BEST_SETTING is True:
#     out_dir = Path("final_result")
#     out_dir.mkdir(exist_ok=True, parents=True)


def drop_ft_keys(setting: dict) -> dict:
    """
    ParameterGrid에서 생성된 setting dict에서
    fine-tune 관련 key만 제거하고 반환합니다.
    """
    ft_keys = {
        "ft_use_es",
        "ft_lr",
        "ft_weight_decay",
        "ft_epoch",
        "ft_head_only",
        "reg_mode",
        "sp_reg",
        "sp_reg_coeff",
        "gc_rate",
        "mb_rate",
        "lora",
    }

    cfg = dict(setting)  # 원본 보존
    for k in list(cfg.keys()):
        if k in ft_keys:
            cfg.pop(k)
    return cfg


def drop_ft_keys_from_list(settings_list: List[dict]) -> List[dict]:
    return [drop_ft_keys(s) for s in settings_list]


# 1. 설정값 검증
def train_model(
    hyper_setting,
    out_dir: Path,
    meta_config,
    train_df_dict,
    valid_df_dict,
    test_df_dict,
    cv_folds_dict=None,
    settings_list=None,
    verbose=False,
):  # settings_list만 학습에 이용
    # Build the iterable of settings (either override list or full grid)
    assert settings_list is not None
    # TODO: modify codes to use settings_list directly without safe_my_param_grid
    settings_iter = (
        list(safe_my_param_grid_with_modification(hyper_setting, meta_config))
        if settings_list is None
        else list(settings_list)
    )  # 학습에 이용될 설정값 리스트
    n_settings = len(settings_iter)

    out_dir.mkdir(exist_ok=True)  # 폴더 없으면 생성/있으면 무시

    existing_files = []
    config_list = []
    for file in my_tqdm(list(out_dir.glob("*.json")), desc="Loading results"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                dic = json.load(f)
            existing_files.append(dic)
            config_list.append(extract_config(dic))
        except Exception as e:
            print(f"[경고] 결과 파일 로드 실패: {file} ({e})")
        if stop_requested():
            raise Exception(f"STOP 요청 감지됨: {STOP_FILE_PATH}")

    pre_dir = Path("pretrain")
    pre_dir.mkdir(exist_ok=True)
    pre_config_dir = pre_dir / "config"
    pre_config_dir.mkdir(parents=True, exist_ok=True)
    pre_existing_files = []
    for file in my_tqdm(
        list(pre_config_dir.glob("*.json")), desc="Loading pretrain results"
    ):
        try:
            with open(file, "r", encoding="utf-8") as f:
                dic = json.load(f)
            # pre_existing_files.append(dic)
            pre_existing_files.append({**dic, "filename": file.name})
        except Exception as e:
            print(f"[경고] config 파일 로드 실패: {file} ({e})")
        if stop_requested():
            raise Exception(f"STOP 요청 감지됨: {STOP_FILE_PATH}")

    print(f"총 {len(pre_existing_files)}개 config 불러옴")

    pre_c_m = ConfigManager(pre_existing_files)
    c_m = ConfigManager(existing_files)

    controls_features_dict, cases_features_dict = run_preprocess()

    # 아래 3개 딕셔너리 키: 'G', 'C', 'E', 'L' ,'P
    # train/val/test 로 구분
    # cases_train_dict = cases_features_dict["train"]
    # cases_val_dict = cases_features_dict["val"]
    # cases_test_dict = cases_features_dict["test"]
    # controls_train_dict = controls_features_dict["train"]
    # controls_val_dict = controls_features_dict["val"]
    # controls_test_dict = controls_features_dict["test"]

    # -------------------- model training -------------------------
    num = RANK
    pre_num = RANK
    for setting in my_tqdm(
        settings_iter, total=n_settings, desc="Iterating hyperparameter settings"
    ):

        # If STOP is requested *before* starting the next experiment, halt here
        if stop_requested():
            raise Exception(f"STOP 요청 감지됨: {STOP_FILE_PATH}")
        if setting["model"] == "Transfer":
            pre_setting = drop_ft_keys(setting)

            pre_matched_dict_list = pre_c_m.find(pre_setting)

            # 동일한 실험 결과가 2개 이상 저장되는 오류가 없는지 확인
            assert (
                len(pre_matched_dict_list) <= 1
            ), f"동일한 실험 결과가 2개 이상 존재합니다: {[dic["exp_name"] for dic in pre_matched_dict_list]}"

            if len(pre_matched_dict_list) == 1:  # 5. 일치하는 결과가 하나라도 있었다면,
                exist_pre_filename = pre_matched_dict_list[0]["filename"].replace(
                    ".json", ""
                )
                if verbose:
                    print(f"Pretrain config found for setting: {pre_setting}")
                    print(f"  Using existing pretrain config: {exist_pre_filename}")
            else:

                assert (
                    DO_PRETRAIN is True
                ), f"Pretrain config not found, but --pretrain flag not set. pre_setting: {pre_setting}"

                # 위에서 정의한 단일 실험 함수 호출
                while True:
                    filename = f"{pre_num}.json"
                    if os.path.exists(pre_config_dir / filename):
                        pre_num += WORLD_SIZE
                    else:  # 존재하지 않는 파일명 찾으면 루프 탈출
                        break

                results, pre_filename = run_single_experiment(
                    setting=setting,
                    out_dir=out_dir,
                    pre_dir=pre_dir,
                    device=device,
                    filename=None,
                    do_only_pretrain=True,
                    load_pretrained_model=False,
                    pre_setting=pre_setting,
                    pre_filename=pre_num,
                    load_data=True,
                    controls_features_dict=controls_features_dict,
                    cases_features_dict=cases_features_dict,
                )
                exist_pre_filename = Path(pre_filename).stem

                # continue
                pre_c_m.append(
                    {
                        **pre_setting,
                        "filename": f"{exist_pre_filename}",
                    }
                )

        if DO_PRETRAIN is False:
            matched_dict_list = c_m.find(setting)

            # 동일한 실험 결과가 2개 이상 저장되는 오류가 없는지 확인
            assert (
                len(matched_dict_list) <= 1
            ), f"동일한 실험 결과가 2개 이상 존재합니다: {[dic["exp_name"] for dic in matched_dict_list]}"

            if len(matched_dict_list) == 1:  # 5. 일치하는 결과가 하나라도 있었다면,
                continue  # 6. 현재 실험을 건너뛰고 다음 setting으로 바로 넘어감

            while True:
                filename = get_file_name(num)
                if os.path.exists(get_file_path(out_dir, num)):
                    num += WORLD_SIZE
                else:  # 존재하지 않는 파일명 찾으면 루프 탈출
                    break
            print(f"Starting experiment {filename} for setting: {setting}")
            # 위에서 정의한 단일 실험 함수 호출
            other_kwargs = {}
            if setting["model"] == "Transfer":
                other_kwargs.update(
                    {
                        "do_only_pretrain": False,
                        "pre_setting": None,
                        "pre_filename": exist_pre_filename,
                        "load_pretrained_model": True,
                    }
                )
            results = run_single_experiment(
                setting=setting,
                out_dir=out_dir,
                pre_dir=pre_dir,
                load_data=True,
                device=device,
                filename=filename,
                RANK=RANK,
                controls_features_dict=controls_features_dict,
                cases_features_dict=cases_features_dict,
                **other_kwargs,
            )

            c_m.append(results)


# --- Build my sliced grid for this process ---
_full_grid = (
    list(safe_my_param_grid_with_modification(HP_SPACE_LIST, meta_config))
    if not DO_PRETRAIN
    else list(
        safe_my_param_grid_with_modification(
            drop_ft_keys_from_list(HP_SPACE_LIST), meta_config
        )
    )
)

if RANK is None:
    # Single-process mode uses the full grid
    _my_settings = _full_grid
else:
    _my_settings = _full_grid[RANK::WORLD_SIZE]
    # list[start::step] RANK 번째부터 WORLD_SIZE 간격으로(ex: 0, 3, 6...)
print(
    f"[Parallel] Total settings={len(_full_grid)} | My share (rank={RANK})={len(_my_settings)}"
)

train_model(
    HP_SPACE_LIST,
    out_dir,
    meta_config,
    None,
    None,
    None,
    None,
    settings_list=_my_settings,  # training에 이용될 설정값 리스트 전달
)

# # %%
# Only rank 0 (or single-process) runs the merge to avoid duplicate work

if not DO_PRETRAIN and WORLD_SIZE == 1 and stop_requested() is False:
    make_merged_results(out_dir)


# %%
