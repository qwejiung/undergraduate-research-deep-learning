import copy
import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from predcancer.data import prepare_datasets
from predcancer.settings import (DEFAULT_PRE_DIR, RUN_SHAP, USE_CV, g_seed,
                                 out_dir)
from predcancer.train_MLP import Train_MLP
from predcancer.train_others import Train_LR, Train_XGB
from predcancer.transfer_MLP import Transfer_MLP
from predcancer.utils import file_path_to_exp_name, get_device, seed_init


def transform_folds(cv_folds_dict, cv_id):
    train_df_dict = {}
    valid_df_dict = {}
    for k in cv_folds_dict.keys():
        fold_id, train_cv, valid_cv = cv_folds_dict[k][cv_id - 1]
        assert fold_id == cv_id, f"Fold ID mismatch: {fold_id} != {cv_id}"
        train_df_dict[k] = train_cv
        valid_df_dict[k] = valid_cv

    return train_df_dict, valid_df_dict


def run_single_experiment(
    setting: dict,
    out_dir: Path = None,
    pre_dir: Path = Path(DEFAULT_PRE_DIR),
    train_df_dict: dict = None,
    valid_df_dict: dict = None,
    test_df_dict: dict = None,
    cv_folds_dict: dict = None,
    full_dict: dict = None,
    device: torch.device = None,
    filename: str = None,  # 파일 이름을 외부에서 받도록 수정
    do_only_pretrain=False,
    pre_setting: dict = None,
    pre_filename: str = None,
    load_data=False,
    RANK=None,
    verbose=False,
    have_best_model=False,
    controls_features_dict=None,
    cases_features_dict=None,
    load_pretrained_model=False,
):
    """
    하나의 하이퍼파라미터 설정(setting)에 대해 모델을 학습하고 결과를 저장합니다.

    Args:
        setting (dict): 실행할 하이퍼파라미터 설정 (ParameterGrid의 단일 샘플).
        ... (기타 필요한 모든 데이터 및 설정 변수)
        filename (str): 결과를 저장할 파일 이름.
        do_only_pretrain (bool): whether to perform pre-training only for transfer learning.
        load_pretrained_model (bool): whether to load a pretrained model for transfer learning.
        load_data (bool): whether to load data inside the function without using the data of the input argument.
    """
    setting["exp_name"] = file_path_to_exp_name(filename)
    split_seed = setting["split_seed"] if "split_seed" in setting else g_seed
    print(f"Running experiment with setting:\n{setting}")
    if device is None:
        device = get_device(RANK)
    if load_data:
        (
            train_df_dict,
            valid_df_dict,
            test_df_dict,
            cv_folds_dict,
            full_dict,
        ) = prepare_datasets(split_seed, controls_features_dict, cases_features_dict)
        print("scaler 저장 완료 #===================")
    MODEL_NAME = setting["model"]
    # 2. 교차 검증(CV) 사용 시, 현재 fold에 맞는 데이터 선택
    if USE_CV:
        train_df_dict, valid_df_dict = transform_folds(cv_folds_dict, setting["cv_id"])

    # 3. 데이터 복사 및 전처리 (gc_rate, feature drop 등)
    train_df_dict_copy = copy.deepcopy(train_df_dict)
    valid_df_dict_copy = copy.deepcopy(valid_df_dict)
    test_df_dict_copy = copy.deepcopy(test_df_dict)
    full_dict_copy = copy.deepcopy(full_dict)

    if do_only_pretrain is False:
        keep_rate = float(setting["gc_rate"])
        if keep_rate < 1.0:
            dfG = train_df_dict_copy["G"]
            train_df_dict_copy["G"], _ = train_test_split(
                dfG,
                train_size=keep_rate,
                stratify=dfG["label"],
                random_state=split_seed,
                shuffle=True,
            )
            train_df_dict_copy["G"] = train_df_dict_copy["G"].reset_index(drop=True)
    else:
        train_df_dict_copy["G"] = None
        valid_df_dict_copy["G"] = None
        test_df_dict_copy["G"] = None

    # drop_cols_list 로 제거할 컬럼 지정
    drop_cols_list = []  # 제거할 열 리스트
    if "drop_cols" in setting and setting["drop_cols"] is not None:
        drop_cols_list = setting["drop_cols"]
    if len(drop_cols_list) > 0:
        for key in train_df_dict_copy.keys():
            if train_df_dict_copy[key] is not None:
                train_df_dict_copy[key] = train_df_dict_copy[key].drop(
                    columns=drop_cols_list, errors="ignore"
                )
                valid_df_dict_copy[key] = valid_df_dict_copy[key].drop(
                    columns=drop_cols_list, errors="ignore"
                )
                test_df_dict_copy[key] = test_df_dict_copy[key].drop(
                    columns=drop_cols_list, errors="ignore"
                )
                full_dict_copy[key] = full_dict_copy[key].drop(
                    columns=drop_cols_list, errors="ignore"
                )

    # 4. 모델 학습 실행
    seed_init(setting["seed"])
    if MODEL_NAME == "Transfer":
        results = Transfer_MLP(
            train_df_dict_copy["G"],
            valid_df_dict_copy["G"],
            test_df_dict_copy["G"],
            device,
            setting,
            train_df_dict=train_df_dict_copy,
            valid_df_dict=valid_df_dict_copy,
            test_df_dict=test_df_dict_copy,
            max_steps=setting["max_steps"],
            valid_steps=setting["valid_steps"],
            do_only_pretrain=do_only_pretrain,
            load_pretrained_model=load_pretrained_model,
            pre_setting=pre_setting,
            pre_filename=pre_filename,
            RUN_SHAP=RUN_SHAP,
            have_best_model=have_best_model,
        )
    else:
        raise Exception("INVALID MODEL")

    if have_best_model is True:
        return results

    if do_only_pretrain is True:
        (pre_dir / "config").mkdir(parents=True, exist_ok=True)
        save_pre_filename = pre_dir / "config" / f"{pre_filename}.json"
        with open(save_pre_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"✅ pre-training 실험 완료 및 저장: {pre_filename}")

        return results, save_pre_filename

    # 5. 결과에 메타데이터와 하이퍼파라미터 추가
    if True or have_best_model is None:
        results.update(setting)
        results["exp_name"] = file_path_to_exp_name(filename)

    # filename = "Transfer_best_model.json"
    if verbose:
        print(f"Finished the experiment")
        print(f"Results: {results}")

    # 6. 최종 결과 JSON 파일로 저장
    if out_dir is not None and filename is not None:
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"✅ 실험 완료 및 저장: {out_path}")
    # TODO: pre-train 성능을 어떻게 저장할지?
    return results
