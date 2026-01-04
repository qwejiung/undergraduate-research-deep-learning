import os
from copy import deepcopy

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from torch.utils.data import DataLoader

from predcancer.settings import (BATCH_SIZE, DEFAULT_FINAL_DIR, ResK,
                                 from_test_probs, get_batch_size,
                                 get_exp_out_dir)
from predcancer.utils import GCDataset, get_sens_spec_f1
from run_shap import run_shap


def compute_y_pred(model: nn.Module, device, loader, logits_f: callable = None):
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(X)
            if logits_f is not None:
                logits = logits_f(logits)
            p = torch.sigmoid(logits)
            y_true_list.append(y.cpu())
            y_pred_list.append(p.cpu())

    y_true = torch.cat(y_true_list).cpu().numpy().ravel()
    y_pred = torch.cat(y_pred_list).cpu().numpy().ravel()

    return y_true, y_pred


def evaluate(
    model: nn.Module,
    device,
    loader,
    thres=None,
    thres_f1=None,
    logits_f: callable = None,
):
    y_true, y_pred = compute_y_pred(model, device, loader, logits_f)

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    sensitivity, specificity, f1, thres, thres_f1 = get_sens_spec_f1(
        y_true, y_pred, thres, thres_f1
    )
    brier = brier_score_loss(y_true, y_pred)
    return (
        auc,
        ap,
        sensitivity,
        specificity,
        f1,
        thres,
        thres_f1,
        {ResK.PROB: y_pred},
        brier,
    )


def Train_MLP(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device,
    setting: dict,
    use_es,
    RUN_SHAP=False,
    verbose=False,
    have_best_model=False,
    final_dir: str = DEFAULT_FINAL_DIR,
):
    exp_name = setting["exp_name"]
    if verbose:
        if use_es:
            print("Using Early Stopping")
        else:
            print("Training without Early Stopping")

    train_ds, val_ds, test_ds = map(GCDataset, (train_df, valid_df, test_df))
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

    # -------------------------- MLP Model ----------------------------------------
    def build_mlp(in_dim, layers, dropout=0.2):
        """layers: e.g., [128, 64] or [64, 64] or [64]"""
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
        mods += [nn.Linear(last, 1)]
        return nn.Sequential(*mods)

    class MLP(nn.Module):
        def __init__(self, in_dim, layers):
            super().__init__()
            self.net = build_mlp(in_dim, layers, setting["dropout"])

        def forward(self, x):
            return self.net(x)

    model = MLP(train_ds.X.shape[1], setting["layer"]).to(device)
    print("model structure:")
    print(model)

    pos_w = torch.tensor(
        [float(setting["pos_weight"])], dtype=torch.float32, device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optim = torch.optim.Adam(
        model.parameters(), lr=setting["lr"], weight_decay=setting["weight_decay"]
    )

    # -------------------------- Training Loop ------------------------------------
    best_val_auc = -1.0
    best_epoch = -1
    val_metrics = {}
    for epoch in range(1, setting["epoch"] + 1):
        model.train()
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()

        (
            val_auc,
            val_ap,
            val_sensitivity,
            val_specificity,
            val_f1,
            val_threshold,
            val_thres_f1,
            _,
            _,
        ) = evaluate(model, device, val_loader)
        val_metrics = {
            "val_auroc": val_auc,
            "val_ap": val_ap,
            "val_sensitivity": val_sensitivity,
            "val_specificity": val_specificity,
            "val_f1": val_f1,
        }
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            # << 수정: 최고 성능일 때 모든 지표 저장
            best_val_metrics = val_metrics
            best_val_thres = val_threshold
            best_val_thres_f1 = val_thres_f1
        if verbose:
            print(f"Epoch {epoch:02d}  val-AUROC={val_auc:.3f}  val-PR={val_ap:.3f}")
    # -------------------------- Final Evaluation ----------------------------------
    if use_es:
        model.load_state_dict(best_state)
        val_metrics = best_val_metrics
        val_threshold = best_val_thres
        val_thres_f1 = best_val_thres_f1
    else:
        best_epoch = setting["epoch"]

    if have_best_model is False:
        os.makedirs(final_dir, exist_ok=True)
        weight_path = os.path.join(final_dir, "MLP_best_model.pt")
        torch.save(model.state_dict(), weight_path)
        print(f"Model weights saved to: {weight_path}")

    y_test, y_pred_prob_test = compute_y_pred(model, device, test_loader)
    results = {
        **val_metrics,  # Validation 결과
        "best_epoch": best_epoch,
        **from_test_probs(
            exp_name, y_pred_prob_test, y_test, val_threshold, val_thres_f1
        ),
    }

    if RUN_SHAP:
        run_shap(
            model,
            device,
            train_df=train_df,
            test_df=test_df,
            save_dir=get_exp_out_dir(exp_name),
        )

    return results
