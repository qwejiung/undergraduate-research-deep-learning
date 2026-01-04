import json
import math
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from predcancer.settings import (BATCH_SIZE, DEFAULT_FINAL_DIR,
                                 DEFAULT_PRE_DIR,
                                 SAVE_DATA_SPLIT_DF_FOR_MODEL_INPUT,
                                 from_test_probs, get_batch_size,
                                 get_exp_out_dir,
                                 get_final_transfer_model_path,
                                 get_infer_prob_path,
                                 get_pretrained_model_path,
                                 get_test_df_csv_path, get_train_df_csv_path,
                                 get_trained_model_path)
from predcancer.train_MLP import compute_y_pred, evaluate
from predcancer.utils import GCDataset, InfiniteDataLoader
from run_shap import run_shap


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 4,
        alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None

        # 기존 가중치/바이어스 복사 & 고정
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
            if self.has_bias
            else None
        )

        # LoRA 파라미터 (ΔW = B @ A,  B: [out, r], A: [r, in])
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        # LoRA dropout
        self.lora_dropout = (
            nn.Dropout(lora_dropout)
            if lora_dropout and lora_dropout > 0
            else nn.Identity()
        )

        # 권장 초기화: A=kaiming_uniform, B=zero (초기 ΔW≈0)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # base
        out = F.linear(x, self.weight, self.bias)
        # lora
        x_d = self.lora_dropout(x)
        # (x @ A^T) @ B^T  ==  x @ (A^T @ B^T)
        lora_out = (x_d @ self.A.t()) @ self.B.t()
        return out + self.scaling * lora_out

    @torch.no_grad()
    def merge_weights_(self):
        # W <- W + scaling * (B @ A)
        delta = self.B @ self.A  # [out, in]
        self.weight.add_(self.scaling * delta)
        # LoRA 경로 비활성화
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)
        self.forward = lambda x: F.linear(x, self.weight, self.bias)


def loralize_backbone(
    model: nn.Module, r=4, alpha=16, lora_dropout=0.0, exclude_last=True
):
    """
    model.net 안의 선형층을 LoRA로 감쌉니다. (마지막 헤드는 기본값으로 제외)
    """
    # nn.Sequential 이므로 인덱스 접근 가능
    last_idx = len(model.net) - 1
    for i, m in enumerate(model.net):
        is_last_linear = (i == last_idx) and isinstance(m, nn.Linear)
        if isinstance(m, nn.Linear) and not (exclude_last and is_last_linear):
            # 교체
            model.net[i] = LoRALinear(m, r=r, alpha=alpha, lora_dropout=lora_dropout)

    # BN은 동결 & eval
    for m in model.net:
        if isinstance(m, nn.BatchNorm1d):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()


def get_trainable_params_for_lora(model: nn.Module, ft_head_only=False):
    """
    - ft_head_only=True: 기존 로직대로 헤드만 학습 (LoRA도 안 씀)
    - ft_head_only=False: 헤드 + LoRA 파라미터(A,B)만 학습
    """
    params = []
    # 헤드 파라미터는 계속 학습
    for p in model.net[-1].parameters():
        p.requires_grad = True
        params.append(p)

    if not ft_head_only:
        # LoRA 파라미터(A,B)만 학습
        for m in model.net:
            if isinstance(m, LoRALinear):
                m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)
                params.extend([m.A, m.B])
    return params


@torch.no_grad()
def merge_all_lora_(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_weights_()


# -------------------------- MLP Model ----------------------------------------
def build_mlp(setting: dict, in_dim, layers, dropout=0.2, ctype_num=1):
    mods = []
    last = in_dim
    for h in layers:
        mods += [
            nn.Linear(last, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        last = h
    if setting["head_mode"] == "new":
        mods += [nn.Linear(last, ctype_num + 1)]
    elif setting["head_mode"] in ["best", "avg"]:
        mods += [nn.Linear(last, ctype_num)]
    else:
        raise Exception(f"Unknown head_mode: {setting['head_mode']}")

    return nn.Sequential(*mods)


class MLP(nn.Module):
    def __init__(self, setting: dict, in_dim, layers, ctype_num=1):
        super().__init__()
        self.net = build_mlp(setting, in_dim, layers, setting["dropout"], ctype_num)

    def forward(self, x):
        return self.net(x)


def get_final_logits(
    setting: dict, logits: torch.Tensor, head_idx=None, eps: float = 1e-7
):
    assert logits.ndim == 2
    if head_idx is not None:
        return logits[:, [head_idx]]  # shape: (B, 1)
    elif setting["head_mode"] == "avg":
        assert head_idx is None
        # p: (B, num_heads)
        p = torch.sigmoid(logits)
        # p_mean: (B, 1)
        p_mean = p.mean(dim=-1, keepdim=True)
        # # numerical stability before logit
        # p_mean = p_mean.clamp(min=eps, max=1 - eps)
        # logit(p) = log(p) - log(1 - p)
        return torch.log(p_mean + eps) - torch.log1p(-p_mean + eps)
    return logits


def make_anchor(
    setting: dict, model: nn.Module, device, sp_reg: bool, mode: str, verbose=False
):
    # requires_grad 여부는 FT 직전에 결정되므로 FT 직후 다시 걸러줄 수도 있음
    anchor = {}
    for name, module in model.named_modules():
        if mode == "all" or (mode == "weight" and isinstance(module, nn.Linear)):
            for pname, p in module.named_parameters(recurse=False):
                if (
                    mode == "weight" and pname == "weight"
                ) or mode == "all":  # bias 제외
                    if sp_reg is True:
                        if setting["head_mode"] in ["avg", "best"]:
                            anchor[f"{name}.{pname}"] = p.detach().clone().to(device)
                        else:
                            if not (module is model.net[-1]):
                                anchor[f"{name}.{pname}"] = (
                                    p.detach().clone().to(device)
                                )
                            else:
                                anchor[f"{name}.{pname}"] = 0.0
                    else:
                        anchor[f"{name}.{pname}"] = 0.0
    if verbose:
        for name, v in anchor.items():
            if isinstance(v, float) and v == 0.0:
                print(f"{name} anchor is zero")
            else:
                print(f"{name} anchor is not zero, shape: {v.shape}")
    return anchor


def l2_anchor_regularization(
    setting,
    model,
    anchor_dict,
    only_trainable=True,
):
    """
    L2-SP toward anchor.
    """
    reg = 0.0
    for name, p in model.named_parameters():
        if only_trainable and not p.requires_grad:
            continue
        if name not in anchor_dict:
            continue
        anchor_v = anchor_dict[name]
        reg_term = torch.sum((p - anchor_v) ** 2)
        if isinstance(anchor_v, float) and anchor_v == 0.0:
            reg += setting["ft_weight_decay"] * reg_term
        else:
            reg += setting["sp_reg_coeff"] * reg_term
    return reg


def Transfer_MLP(
    train_df_ft,
    valid_df_ft,
    test_df_ft,
    device,
    setting: dict,
    train_df_dict=None,
    valid_df_dict=None,
    test_df_dict=None,
    max_steps: int = None,
    valid_steps: int = None,
    do_only_pretrain: bool = False,
    pre_dir=Path(DEFAULT_PRE_DIR),
    pre_setting=None,
    pre_filename=None,
    RUN_SHAP=False,
    verbose=False,
    have_best_model=False,
    final_dir: str = DEFAULT_FINAL_DIR,
    load_pretrained_model=False,
):
    exp_name = setting["exp_name"]
    exp_dir = get_exp_out_dir(exp_name)
    # TODO: deprecate `load_data`
    pre_use_es = setting["pre_use_es"]
    pre_epoch = setting.get("pre_epoch")
    if do_only_pretrain is False:
        ft_use_es = setting["ft_use_es"]
    c_types = setting["cancer_type"]

    if train_df_ft is not None:
        print(f"train_df_ft shape: {train_df_ft.shape}")

    def make_loaders(train_df, valid_df, test_df, device, pretrain=True):
        train_ds, val_ds, test_ds = map(GCDataset, (train_df, valid_df, test_df))
        print(f"train_ds size: {len(train_ds)}")
        train_loader = DataLoader(
            train_ds,
            batch_size=(
                BATCH_SIZE
                if setting.get("mb_rate") is None
                else get_batch_size(len(train_ds), setting["mb_rate"])
            ),
            shuffle=True,
            drop_last=setting.get("drop_last", False),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=256,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=256,
        )
        if pretrain:
            return train_ds, train_loader, val_loader, test_loader
        else:
            return train_loader, val_loader, test_loader

    # model head 암 개수만큼
    # 현재 forward는 모든 암에 대한 결과가 나온다. 그 중에 어떤 score를 쓸지 선택해야 함.
    ctype_to_idx: dict[str, int] = {
        c: i for i, c in enumerate(c_types)
    }  # ex) {'P':0,'C':1}
    if setting["head_mode"] == "new":
        ctype_to_idx["G"] = len(c_types)  # 신규 head에 control용 추가
    print("c_types:", c_types)

    train_ds_pre_dict = {}
    train_loader_pre_dict = {}
    val_loader_pre_dict = {}
    test_loader_pre_dict = {}
    if setting["use_epoch_loader"] and len(c_types) == 1:
        ctype = c_types[0]
        train_ds_pre, train_loader_pre, val_loader_pre, test_loader_pre = make_loaders(
            train_df_dict[ctype],
            valid_df_dict[ctype],
            test_df_dict[ctype],
            device,
            True,
        )
    else:
        train_ds_pre = None
        train_loader_pre = None
        val_loader_pre = None
        test_loader_pre = None
        for ctype in c_types:
            (
                train_ds_pre_dict[ctype],
                train_loader_pre_dict[ctype],
                val_loader_pre_dict[ctype],
                test_loader_pre_dict[ctype],
            ) = make_loaders(
                train_df_dict[ctype],
                valid_df_dict[ctype],
                test_df_dict[ctype],
                device,
                True,
            )
            train_loader_pre_dict[ctype] = InfiniteDataLoader(
                train_loader_pre_dict[ctype]
            )
    if do_only_pretrain is False:
        train_ds_ft, val_ds_ft, test_ds_ft = map(
            GCDataset, (train_df_ft, valid_df_ft, test_df_ft)
        )
        train_loader_ft, val_loader_ft, test_loader_ft = make_loaders(
            train_df_ft, valid_df_ft, test_df_ft, device, False
        )

    if setting["use_epoch_loader"] and len(c_types) == 1:
        input_dim = train_ds_pre.X.shape[1]
    else:
        input_dim = [train_ds_pre_dict[ctype].X.shape[1] for ctype in c_types]
        assert all(
            [input_dim[0] == d for d in input_dim]
        ), "Input dimensions do not match among cancer types"
        input_dim = input_dim[0]

    model = MLP(setting, input_dim, setting["layer"], len(c_types)).to(device)

    if verbose:
        print("model structure:")
        print(model)

    pos_w = torch.tensor(
        [float(setting["pos_weight"])], dtype=torch.float32, device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def make_optimizer(params, lr, weight_decay):
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def evaluate_multi(val_loader_dict, c_types=None):
        """각 type별 성능을 계산하고 평균을 반환"""
        aucs, aps = [], []
        sensitivity_list, specificity_list, f1_list = [], [], []
        for ctype in c_types:
            idx = ctype_to_idx[ctype]
            auc, ap, sensitivity, specificity, f1, _, _, _, _ = evaluate(
                model,
                device,
                val_loader_dict[ctype],
                logits_f=lambda _logits: get_final_logits(setting, _logits, idx),
            )
            aucs.append(auc)
            aps.append(ap)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            f1_list.append(f1)
        return (
            mean(aucs),
            mean(aps),
            mean(sensitivity_list),
            mean(specificity_list),
            mean(f1_list),
        )

    def pick_best_type(val_loader_dict, c_types):
        """각 타입의 val AUROC/AP를 계산하고, AUROC(1st)→AP(2nd)로 가장 좋은 타입을 반환"""
        per_metrics = []  # (ctype, auc, ap)
        for ctype in c_types:
            idx = ctype_to_idx[ctype]
            auc, ap, sensitivity, specificity, f1, _, _, _, _ = evaluate(
                model,
                device,
                val_loader_dict[ctype],
                logits_f=lambda _logits: get_final_logits(setting, _logits, idx),
            )
            per_metrics.append((ctype, auc, ap))
        # AUROC 우선, 동률이면 AP로 비교
        per_metrics.sort(key=lambda x: (x[1], x[2]))
        best_type = per_metrics[-1][0]
        return best_type, per_metrics

    def Training_LOOP(
        use_es: bool,
        max_epoch,
        train_loader,
        val_loader,
        device,
        model,
        optim,
        loss_fn,
        train_loader_pre_dict,
        val_loader_pre_dict,
        c_types: List[str] = None,
        ft_c_types: List[str] = None,
        anchor=None,
        pretrain=None,
    ):
        assert pretrain is not None
        # B: batch size
        if pretrain is False:  # finetune
            assert c_types is None
            if setting["head_mode"] == "avg":
                assert ft_c_types is None
            else:
                assert ft_c_types is not None and len(ft_c_types) == 1
        else:
            assert c_types == setting["cancer_type"]
            assert ft_c_types is None

        use_types: List[str] = ft_c_types if ft_c_types is not None else c_types
        if pretrain is False and setting["head_mode"] == "avg":
            assert use_types is None

        if pretrain is False:  # finetune
            use_epoch_loader = True
        else:
            use_epoch_loader = setting["use_epoch_loader"]
        assert isinstance(use_epoch_loader, bool)

        if use_epoch_loader is True:
            head_idx = None
            if use_types is not None and len(use_types) == 1:
                head_idx = ctype_to_idx[use_types[0]]

            val_auc = -1.0
            best_epoch = -1
            best_state = None
            for epoch in range(1, max_epoch + 1):
                model.train()
                for X, y in train_loader:
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optim.zero_grad()
                    pred = model(X)  # shape: (B, #(cancer types))
                    pred = get_final_logits(setting, pred, head_idx)
                    loss = loss_fn(pred, y)

                    if anchor is not None and (
                        setting["ft_weight_decay"] > 0 or setting["sp_reg_coeff"] > 0
                    ):
                        loss = loss + l2_anchor_regularization(
                            setting,
                            model,
                            anchor,
                            only_trainable=True,
                        )

                    loss.backward()
                    optim.step()

                (
                    last_val_auc,
                    last_val_ap,
                    last_val_sensitivity,
                    last_val_specificity,
                    last_val_f1,
                    last_val_thres,
                    last_val_thres_f1,
                    _,
                    _,
                ) = evaluate(
                    model,
                    device,
                    val_loader,
                    logits_f=lambda _logits: get_final_logits(
                        setting, _logits, head_idx
                    ),
                )

                if last_val_auc > val_auc:  # best_auc
                    val_auc = last_val_auc
                    val_ap = last_val_ap
                    val_sensitivity = last_val_sensitivity
                    val_specificity = last_val_specificity
                    val_f1 = last_val_f1
                    val_thres = last_val_thres
                    val_thres_f1 = last_val_thres_f1
                    best_epoch = epoch
                    best_state = deepcopy(model.state_dict())

                if verbose:
                    print(
                        f"Epoch {epoch:02d}  val-AUROC={val_auc:.3f}  val-PR={val_ap:.3f}"
                    )
            if use_es:
                return (
                    val_auc,
                    val_ap,
                    val_sensitivity,
                    val_specificity,
                    val_f1,
                    best_state,
                    best_epoch,
                    val_thres,
                    val_thres_f1,
                )
            else:
                return (
                    last_val_auc,
                    last_val_ap,
                    last_val_sensitivity,
                    last_val_specificity,
                    last_val_f1,
                    None,
                    None,
                    last_val_thres,
                    last_val_thres_f1,
                )

        else:  # not use_epoch_loader

            val_auc, best_step, best_state = -1.0, -1, None
            for step in range(max_steps):
                model.train()
                optim.zero_grad()
                total_loss = 0.0
                for ctype in use_types:  # 1개 or 2개
                    head_idx = ctype_to_idx[ctype]
                    X, y = next(train_loader_pre_dict[ctype])
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    pred = model(X)  # (batch, 암타입개수)
                    pred_head = pred[:, [head_idx]]  # (B,1)
                    loss = loss_fn(pred_head, y)
                    total_loss += loss
                    # loss = loss_fn(pred, y)
                    # total_loss += loss
                total_loss.backward()
                optim.step()
                # sigmoid, probability 쓰게되면 다르게 해야함

                if (step + 1) % valid_steps == 0:
                    (
                        last_val_auc,
                        last_val_ap,
                        last_val_sensitivity,
                        last_val_specificity,
                        last_val_f1,
                    ) = evaluate_multi(val_loader_pre_dict, c_types=c_types)

                    if use_es and (last_val_auc > val_auc):
                        val_auc = last_val_auc
                        val_ap = last_val_ap
                        val_sensitivity = last_val_sensitivity
                        val_specificity = last_val_specificity
                        val_f1 = last_val_f1
                        best_step = step
                        best_state = deepcopy(model.state_dict())

                    # print(
                    #     f"[Step {step+1:04d}] val-AUROC={val_auc:.3f}  val-AP={val_ap:.3f}"
                    # )

            if use_es and best_state is not None:
                return (
                    val_auc,
                    val_ap,
                    val_sensitivity,
                    val_specificity,
                    val_f1,
                    best_state,
                    best_step,
                )
            else:
                return (
                    last_val_auc,
                    last_val_ap,
                    last_val_sensitivity,
                    last_val_specificity,
                    last_val_f1,
                    None,
                    None,
                )

    # -------------------------- Final Evaluation ----------------------------------

    def enable_only_head(model, head_idx: int):
        last_linear = list(model.net.children())[-1]  # nn.Linear(last, ctype_num)

        # weight: (ctype_num, last), bias: (ctype_num)
        w_mask = torch.zeros_like(last_linear.weight)
        b_mask = torch.zeros_like(last_linear.bias)
        w_mask[head_idx : head_idx + 1, :] = 1
        b_mask[head_idx] = 1

        # 기존 hook 제거 대비: 핸들을 저장하고 필요시 .remove() 하면 됨
        def w_hook(grad):
            return grad * w_mask

        def b_hook(grad):
            return grad * b_mask

        # requires_grad는 켜둬야 hook이 효과가 있음
        last_linear.weight.requires_grad = True
        last_linear.bias.requires_grad = True
        h1 = last_linear.weight.register_hook(w_hook)
        h2 = last_linear.bias.register_hook(b_hook)
        return (h1, h2)  # 나중에 필요하면 .remove()로 해제

    if have_best_model is False:
        (pre_dir / "data").mkdir(parents=True, exist_ok=True)
        (pre_dir / "result").mkdir(parents=True, exist_ok=True)

        if not load_pretrained_model:
            # pretraining
            pre_optim = make_optimizer(
                model.parameters(), setting["pre_lr"], setting["pre_weight_decay"]
            )
            (
                pre_val_auc,
                pre_val_ap,
                pre_val_sensitivity,
                pre_val_specificity,
                pre_val_f1,
                pre_best_state,
                pre_best_epoch,
            ) = Training_LOOP(
                pre_use_es,
                pre_epoch,
                train_loader_pre,
                val_loader_pre,
                device,
                model,
                pre_optim,
                loss_fn,
                train_loader_pre_dict,
                val_loader_pre_dict,
                c_types,
                ft_c_types=None,
                anchor=None,
                pretrain=True,
            )

            if pre_use_es:
                model.load_state_dict(pre_best_state)
            pre_results = {
                "pre_val_auroc": pre_val_auc,
                "pre_val_ap": pre_val_ap,
                "pre_val_sensitivity": pre_val_sensitivity,
                "pre_val_specificity": pre_val_specificity,
                "pre_val_f1": pre_val_f1,
                "pre_best_step_or_epoch": pre_best_epoch,
            }
            if do_only_pretrain:
                result_pre_filename = pre_dir / "result" / f"{pre_filename}.json"
                with open(result_pre_filename, "w", encoding="utf-8") as f:
                    json.dump(pre_results, f, indent=4, ensure_ascii=False)

                pre_model_path = get_pretrained_model_path(pre_dir, pre_filename)
                torch.save(model.state_dict(), pre_model_path)
                print(f"Pretrained model saved to {pre_model_path}")
                return pre_setting
        else:
            assert pre_dir is not None and pre_filename is not None
            weight_path = get_pretrained_model_path(pre_dir, pre_filename)
            state = torch.load(weight_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded pretrained model from {weight_path}")

        anchor = make_anchor(
            setting,
            model,
            device,
            setting["sp_reg"],
            mode=setting["reg_mode"],
            verbose=verbose,
        )

        def get_finetune_types() -> List[str] | None:
            if setting["head_mode"] == "new":
                finetune_types = ["G"]
            elif setting["head_mode"] == "best":
                if len(c_types) > 1:
                    selected_type, per_metrics = pick_best_type(
                        val_loader_pre_dict, c_types
                    )
                    print("Per-type validation after pretraining:")
                    for c, auc, ap in per_metrics:
                        print(f"  {c}: AUROC={auc:.4f}, AP={ap:.4f}")
                    print(f"=> Selected type for fine-tuning: {selected_type}")
                    finetune_types = [selected_type]
                else:
                    finetune_types = c_types
            elif setting["head_mode"] == "avg":
                finetune_types = None
            else:
                raise Exception(f"Unknown head_mode: {setting['head_mode']}")
            return finetune_types

        finetune_types = get_finetune_types()

        if len(c_types) == 1 and setting["use_epoch_loader"]:
            head_idx = ctype_to_idx[c_types[0]]  # 'new'여도 이 인덱스는 해당 ctype head
            (
                pre_test_auc,
                pre_test_ap,
                pre_test_sensitivity,
                pre_test_specificity,
                pre_test_f1,
                _,
                _,
                _,
                _,
            ) = evaluate(
                model,
                device,
                test_loader_pre,
                logits_f=lambda _logits: get_final_logits(setting, _logits, head_idx),
            )
        else:
            (
                pre_test_auc,
                pre_test_ap,
                pre_test_sensitivity,
                pre_test_specificity,
                pre_test_f1,
            ) = evaluate_multi(test_loader_pre_dict, c_types)
        if verbose:
            print(f"Pre_Test AUROC={pre_test_auc:.3f}  |  Pre_PR‑AUC={pre_test_ap:.3f}")

        # trasfer learning

        # === Freeze backbone & prepare FT ===
        last_linear = model.net[-1]  # 마지막 Linear 레이어
        keep_params = set(p for p in last_linear.parameters())

        if setting["ft_head_only"]:
            # 1) 마지막 Linear를 제외한 모든 파라미터 freeze
            for p in model.parameters():
                if p not in keep_params:
                    p.requires_grad = False

            # 2) BN/Dropout을 eval 모드로 고정 (backbone 쪽만)
            model.net[:-1].train(False)
            for m in model.net[:-1].modules():
                if isinstance(m, nn.BatchNorm1d):
                    for q in m.parameters():
                        q.requires_grad = False  # BN도 학습 막기
        else:
            # ft_head_only=False면 백본도 학습
            for p in model.parameters():
                p.requires_grad = True
            model.train(True)  # Dropout/BN 포함 전체 학습 모드

        hook_handles = None
        ft_head_idx = None
        if finetune_types is not None:
            ft_head_idx = ctype_to_idx[finetune_types[0]]
            hook_handles = enable_only_head(model, ft_head_idx)

        if setting["lora"] is True:
            loralize_backbone(model, r=1, alpha=4, lora_dropout=0.0, exclude_last=True)
            model = model.to(device)

        # 옵티마이저는 마지막 Linear만 넘겨도 좋고, 전체 파라미터를 넘겨도
        # backbone은 requires_grad=False라 업데이트되지 않고,
        # 마지막 Linear도 hook 덕분에 해당 head row만 업데이트됨.
        if setting["lora"] is True:
            assert (
                setting["ft_head_only"] is False
            ), "LoRA with head-only FT is not allowed"
            trainable = get_trainable_params_for_lora(
                model, ft_head_only=setting["ft_head_only"]
            )  # 헤드 + LoRA 학습
            ft_optim = torch.optim.Adam(
                trainable, lr=setting["ft_lr"], weight_decay=0.0
            )
        else:
            ft_optim = torch.optim.Adam(
                (
                    list(model.net[-1].parameters())
                    if setting["ft_head_only"]
                    else model.parameters()
                ),
                lr=setting["ft_lr"],
            )

        if verbose:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    if (
                        n in anchor
                        and isinstance(anchor[n], float)
                        and anchor[n] == 0.0
                    ):
                        print(f"[FT] {n} is trainable but anchor is zero")
                    else:
                        print(f"[FT] {n} is trainable")
                else:
                    print(f"[FT] {n} is frozen")
        # -------------------- Measure fine-tuning wall time --------------------
        _t0_ft = time.perf_counter()
        (
            ft_val_auc,
            ft_val_ap,
            ft_val_sensitivity,
            ft_val_specificity,
            ft_val_f1,
            ft_best_state,
            ft_best_epoch,
            ft_val_thres,
            ft_val_thres_f1,
        ) = Training_LOOP(
            ft_use_es,
            setting["ft_epoch"],
            train_loader_ft,
            val_loader_ft,
            device,
            model,
            ft_optim,
            loss_fn,
            train_loader_pre_dict=None,
            val_loader_pre_dict=None,
            c_types=None,
            ft_c_types=finetune_types,
            anchor=anchor,
            pretrain=False,
        )
        ft_elapsed_sec = time.perf_counter() - _t0_ft
        if ft_use_es:
            model.load_state_dict(ft_best_state)

        if hook_handles is not None:
            for h in hook_handles:
                h.remove()
    else:
        weight_path = get_final_transfer_model_path(final_dir)
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        if setting.get("head_mode", "new") == "new":
            ft_head_idx = len(c_types)

    torch.save(model.state_dict(), get_trained_model_path(exp_dir))

    y_test, prob = compute_y_pred(
        model,
        device,
        test_loader_ft,
        lambda _logits: get_final_logits(setting, _logits, ft_head_idx),
    )
    np.savetxt(get_infer_prob_path(final_dir), prob, delimiter=",")
    test_results = from_test_probs(
        exp_name, prob, y_test, ft_val_thres, ft_val_thres_f1
    )

    if have_best_model is True:
        return test_results
    else:  # 이렇게되면 무조건 final_dir에 모델이 저장되는데 모든 애들이 같은 이름으로 저장돼서 사라짐.
        weight_path = get_final_transfer_model_path(final_dir)
        torch.save(model.state_dict(), weight_path)
        print(f"Model weights saved to: {weight_path}")

    # pre-result load
    if load_pretrained_model:
        pre_result_path = pre_dir / "result" / f"{pre_filename}.json"
        _pre_res = {}
        if pre_result_path.exists():
            try:
                with open(pre_result_path, "r", encoding="utf-8") as f:
                    _pre_res = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load pre_result: {pre_result_path} ({e})")
        else:
            print(f"[INFO] No pre_result file: {pre_result_path}")
    else:
        _pre_res = pre_results
    print(f"elapsed time for FT: {ft_elapsed_sec:.1f} sec")
    results = {
        **_pre_res,
        "pre_test_auroc": pre_test_auc,
        "pre_test_ap": pre_test_ap,
        "pre_test_sensitivity": pre_test_sensitivity,
        "pre_test_specificity": pre_test_specificity,
        "pre_test_f1": pre_test_f1,
        "val_auroc": ft_val_auc,
        "val_ap": ft_val_ap,
        "val_sensitivity": ft_val_sensitivity,
        "val_specificity": ft_val_specificity,
        "val_f1": ft_val_f1,
        "ft_val_thres": ft_val_thres,
        "ft_val_thres_f1": float(ft_val_thres_f1),
        **test_results,
        "ft_best_epoch": ft_best_epoch,
        "selected_type": finetune_types[0] if finetune_types is not None else None,
        "ft_elapsed_sec": ft_elapsed_sec,
    }
    if SAVE_DATA_SPLIT_DF_FOR_MODEL_INPUT:
        train_df_ft.to_csv(get_train_df_csv_path(), index=False)
        test_df_ft.to_csv(get_test_df_csv_path(), index=False)
        raise Exception("Data split saved. Stop here.")
    if RUN_SHAP:
        run_shap(
            model,
            device,
            train_df=train_df_ft,
            test_df=test_df_ft,
            save_dir=get_exp_out_dir(exp_name),
        )

    return results
