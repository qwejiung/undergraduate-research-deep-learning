# controls_features_dict, cases_features_dict = run_preprocess()

# full = pd.concat([cases_features_dict["G"], controls_features_dict["G"]], axis=0)
# cases_features_dict: 각 암에 대한 cases 데이터프레임 dict
# controls_features_dict: 각 암 환자에 대해 매칭한 controls 데이터프레임 dict
# 암의 첫 대문자를 key로 접근("G", "C", "L", "E", "P")
# %%
# compare_case_control.py  -----------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from predcancer.settings import PROJECT_DIR

# 0.  Load the dataset ----------------------------------------------------
# If 'full' is already in memory from the earlier notebook, just skip this:
# full = pd.read_pickle("full.pkl")
# %%
# controls_features_dict, cases_features_dict = run_preprocess()


# full = pd.concat([cases_features_dict["G"], controls_features_dict["G"]], axis=0)
# %%
def save_stat(full: pd.DataFrame, cancer_name: str, MD: bool):

    assert "label" in full.columns, "Need the combined DataFrame with a 'label' column."
    cases = full[full.label == 1]  # cases 데이터프레임
    controls = full[full.label == 0]  # controls 데이터프레임

    # 1.  Identify variable types --------------------------------------------
    binary_cols = [  # binary 값(0, 1) 인 열의 이름을 담고 있는 리스트
        c
        for c in full.columns
        if full[c].dropna().isin([0, 1]).all()
        and c not in ("label", "subject_id", "c_time", "admittime_GAP")
    ]
    numeric_cols = [  # numeric 값 인 열의 이름을 담고 있는 리스트 (not binary)
        c
        for c in full.columns
        if c not in binary_cols + ["label", "subject_id", "c_time"]
        and np.issubdtype(full[c].dtype, np.number)
    ]

    # %%

    # 2.  Helper functions ----------------------------------------------------
    def mw_p(x, y):
        """Mann‑Whitney P‑value (two‑sided); returns NaN if all-NaN."""
        if x.dropna().empty or y.dropna().empty:
            return np.nan
        return ss.mannwhitneyu(x, y, alternative="two-sided").pvalue

    # %%
    def chi2_p(x, y):
        """Chi‑square P‑value for 2×2 table; returns NaN if counts too small."""
        tbl = pd.crosstab(x, y)
        if tbl.shape != (2, 2):
            return np.nan
        return ss.chi2_contingency(tbl, correction=False)[1]

    # %%
    # 3.  Collect results -----------------------------------------------------
    rows = []

    for col in numeric_cols:
        cases_filtered = cases[col].dropna()
        controls_filtered = controls[col].dropna()
        rows.append(
            {
                "variable": col,
                "type": "numeric",
                "case_median": cases_filtered.median(),
                "case_IQR": cases_filtered.quantile([0.25, 0.75]).values,
                "ctrl_median": controls_filtered.median(),
                "ctrl_IQR": controls_filtered.quantile([0.25, 0.75]).values,
                "P_value": mw_p(cases_filtered, controls_filtered),
            }
        )

    for col in binary_cols:
        rows.append(
            {
                "variable": col,
                "type": "binary",
                "case_%1": cases[col].mean() * 100,
                "ctrl_%1": controls[col].mean() * 100,
                "P_value": chi2_p(full[col], full["label"]),
            }
        )

    report = pd.DataFrame(rows)
    # cols: variable, type, case_median, case_IQR, ctrl_median, ctrl_IQR, P_value

    CSV_SAVE_DIR = os.path.join(PROJECT_DIR, "csv_report")
    os.makedirs(CSV_SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(
        CSV_SAVE_DIR,
        f"case_control_variable_comparison_CANCER-{cancer_name}_MATCH_DATE-{MD}.csv",
    )
    report.to_csv(csv_path, index=False)
    print(f"✓  Saved summary → {csv_path}")
    # 4.  Save tidy results ---------------------------------------------------
    # report.to_csv("case_control_variable_comparison.csv", index=False)
    # print("✓  Saved summary → case_control_variable_comparison.csv")

    # 5.  (Optional) Make quick plots ----------------------------------------
    # PLOT_DIR = "cc_plots"
    # os.makedirs(PLOT_DIR, exist_ok=True)

    # for col in numeric_cols:
    #     plt.figure(figsize=(4, 4))
    #     sns.boxplot(x="label", y=col, data=full, showfliers=False)
    #     sns.stripplot(
    #         x="label",
    #         y=col,
    #         data=full.sample(frac=0.1, random_state=0),
    #         size=2,
    #         alpha=0.4,
    #         color="k",
    #     )
    #     plt.xticks([0, 1], ["Control", "Case"])
    #     plt.title(col)
    #     plt.tight_layout()
    #     plt.savefig(f"{PLOT_DIR}/{col}_box.png")
    #     plt.close()

    # for col in binary_cols:
    #     plt.figure(figsize=(3, 3))
    #     sns.barplot(x="label", y=col, data=full, estimator=np.mean)
    #     plt.xticks([0, 1], ["Control", "Case"])
    #     plt.ylabel("Proportion = Mean")
    #     plt.title(col)
    #     plt.ylim(0, 1)
    #     plt.tight_layout()
    #     plt.savefig(f"{PLOT_DIR}/{col}_bar.png")
    #     plt.close()

    # print(f"✓  Plots stored in {PLOT_DIR}/")

    # %%
    report
