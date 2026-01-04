from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import torch

from predcancer.settings import TEST_DF_CSV_PATH, TRAIN_DF_CSV_PATH
from predcancer.utils import GCDataset


def run_shap(
    model,
    device,
    model_name="",
    train_df=None,
    test_df=None,
    save_dir=None,
    head_idx=None,
):

    train_csv = TRAIN_DF_CSV_PATH
    test_csv = TEST_DF_CSV_PATH
    if train_df is None:
        train_df = pd.read_csv(train_csv)
    if test_df is None:
        test_df = pd.read_csv(test_csv)

    # ---------------- 2. Dataset 변환 ----------------
    train_ds = GCDataset(train_df)
    test_ds = GCDataset(test_df)

    feature_names = list(test_df.drop(columns=["label"]).columns)

    assert len(feature_names) == test_ds.X.shape[1], "name/vector length mismatch"

    def prettify_feature(name: str) -> str:
        mapping = {
            "RDW": "RDW",
            "MCHC": "MCHC",
            "MCH": "MCH",
            "GERD": "GERD",
            "HGB": "HGB",
        }
        if name in mapping:
            return mapping[name]
        # 기본 규칙: 언더스코어 → 공백, 첫 글자만 대문자
        # 예: "Basophils" → "Basophils", "AGE" → "Age"
        pretty = name.replace("_", " ")
        if pretty.isupper():
            pretty = pretty.title()
        return pretty

    display_names = [prettify_feature(n) for n in feature_names]

    # 2)  pick a small background set (≈1 k rows) for SHAP’s expectations
    bg_idx = torch.randperm(len(train_ds))[:1024]
    # train_ds 전체 인덱스를 무작위로 섞고 1024개 추출
    bg_tensor = train_ds.X[bg_idx].to(device)  # train_ds.X: 학습 데이터의 feature 텐서

    # 3)  choose the subset of test patients you want explanations for
    test_idx = torch.randperm(len(test_ds))[:2000]  # 2 k rows ~= 2 s on GPU
    test_tensor = test_ds.X[test_idx].to(device)

    # ---------------- 1.  Build explainer ----------------------------
    explainer = shap.DeepExplainer(model, bg_tensor)  # 신경망 모델 전용 SHAP 해석기

    # 각 feature가 모델 예측에 얼마나 기여했는지 계산하고 시각화
    shap_results = explainer.shap_values(test_tensor)

    # shap_values is a list with one element (because output dim = 1)
    shap_values = shap_results  # shape (N, D) or (N, D, #(Heads))
    if head_idx is not None:
        shap_values = shap_values[:, :, head_idx]

    plt.rcParams.update(
        {
            "font.size": 10,  # 글자 크기 조정
            "axes.titlesize": 12,  # 제목 크기
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 200,  # 출력 해상도 향상
            "figure.figsize": (6, 5),  # 전체 그림 크기 조정
            # "savefig.bbox": "tight",  # 저장 시 여백 최소화
        }
    )

    X_np = test_tensor.detach().cpu().numpy()

    plt.figure()
    shap.summary_plot(
        shap_values,
        features=X_np,
        feature_names=display_names,
        plot_type="bar",
        max_display=15,  # 필요하면 확대/축소
        show=False,
    )
    # plt.tight_layout()
    ax = plt.gca()
    ax.set_xlabel("Mean(|SHAP value|)")
    if save_dir is None:
        save_dir = Path("tex_and_figs")
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

    file_path = save_dir / f"{model_name}_shap_summary_bar_global"
    plt.savefig(f"{file_path}.pdf", dpi=300)
    plt.savefig(f"{file_path}.png", dpi=300)

    plt.figure()
    shap.summary_plot(
        shap_values, features=X_np, feature_names=display_names, show=False
    )
    # plt.tight_layout()

    file_path = save_dir / f"{model_name}_shap_summary_dot_global"
    plt.savefig(f"{file_path}.pdf", dpi=300)
    plt.savefig(f"{file_path}.png", dpi=300)

    # ---------------- 3.  Per‑patient waterfall ----------------------

    i = 0  # index in the test_idx sample
    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value[0],
            data=test_tensor[i].cpu().numpy(),
            feature_names=display_names,
        ),
        show=False,
        max_display=15,
    )
    plt.title(f"Subject {test_idx[i].item()} – top 15 features")
    # plt.tight_layout()

    file_path = save_dir / f"{model_name}_shap_patient_{i}.png"
    plt.savefig(file_path, dpi=300)
    print(f"Saved individual explanation → shap_patient_{i}.png")
