import json
import os
from pathlib import Path

import pandas as pd

from predcancer.manage_json import ConfigManager
from predcancer.settings import (HP_SPACE_LIST, MERGED_DIR, meta_config,
                                 safe_my_param_grid_with_modification,
                                 stop_requested)
from predcancer.utils import dic_match, get_metric_cols, my_tqdm
from remove_const_col import drop_constant_columns

# You already have these in your env; fallback to current dir if not importable
try:
    from predcancer.settings import MERGED_DIR
except Exception:
    MERGED_DIR = "./merged_results"


# -----------------------------
# Helpers
# -----------------------------
def _latex_escape(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # Minimal escaping for LaTeX tables
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _load_best_csv(group_keys, merged_dir: Path) -> pd.DataFrame:
    """
    Load the 'best_by_{keys}.csv' produced by your existing pipeline.
    """
    suffix = "-".join(group_keys).replace(os.sep, "_")
    path_csv = merged_dir / f"best_by_{suffix}.csv"
    if not path_csv.exists():
        raise FileNotFoundError(f"Not found: {path_csv}")
    return pd.read_csv(path_csv)


def _metric_cols_available(df: pd.DataFrame):
    """
    Auto-detect metric columns to include in tables.
    Priority: test_auroc_mean, test_auprc_mean, val_auroc_mean (if present).
    """
    candidates = [
        "test_auroc_mean",
        "test_auprc_mean",
        "val_auroc_mean",
    ]
    return [c for c in candidates if c in df.columns]


def _format_number(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _to_latex_table(df: pd.DataFrame, caption: str, label: str, index=False) -> str:
    """
    Convert a small, already-prepared DataFrame to a compact LaTeX table.
    """
    # Escape column names
    df2 = df.copy()
    df2.columns = [_latex_escape(str(c)) for c in df2.columns]
    if index and df2.index.name:
        df2.index.name = _latex_escape(str(df2.index.name))
    # Pandas to_latex with booktabs looks clean in papers
    latex = df2.to_latex(
        index=index,
        escape=False,  # we already escaped
        longtable=False,
        bold_rows=False,
        float_format=lambda x: f"{x:.3f}",
        column_format=None,  # let LaTeX decide widths
        buf=None,
    )
    # Add caption/label wrapper
    wrapped = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"{latex}\n"
        f"\\caption{{{_latex_escape(caption)}}}\n"
        f"\\label{{{_latex_escape(label)}}}\n"
        "\\end{table}\n"
    )
    return wrapped


def _safe_save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")
    print(f"✅ LaTeX table saved → {path}")


# -----------------------------
# 1) 각 model 별 성능 비교 (gc_rate==1.0)
# -----------------------------
def make_table_model_compare_gc1(
    merged_dir=MERGED_DIR, out_name="table_model_compare_gc1.tex"
):
    merged_dir = Path(merged_dir)
    df = _load_best_csv(["gc_rate", "model"], merged_dir)
    df = df[df["gc_rate"] == 1.0].copy()
    metrics = _metric_cols_available(df)
    if not metrics:
        raise ValueError(
            "No expected metric columns found in best_by_gc_rate-model.csv"
        )

    # Keep only: model + metrics
    cols = ["model"] + metrics
    df = df[cols].copy()

    # Make pretty: round metrics
    for m in metrics:
        df[m] = df[m].map(lambda v: _format_number(v, 3))

    latex = _to_latex_table(
        df.sort_values(by=metrics[0], ascending=False).reset_index(drop=True),
        caption="Model comparison at $\\mathrm{gc\\_rate}=1.0$ (best-by-validation).",
        label="tab:model-compare-gc1",
        index=False,
    )
    _safe_save_text(latex, merged_dir / out_name)


# -----------------------------
# 2) Transfer: gc_rates × sp_reg 성능 비교
# -----------------------------
def make_table_transfer_spreg(
    merged_dir=MERGED_DIR,
    gc_rates=(1.0, 0.5, 0.2, 0.1, 0.05),
    out_name="table_transfer_spreg.tex",
):
    merged_dir = Path(merged_dir)
    df = _load_best_csv(["gc_rate", "model", "sp_reg"], merged_dir)
    df = df[df["model"] == "Transfer"].copy()
    df = df[df["gc_rate"].isin(gc_rates)].copy()

    metrics = _metric_cols_available(df)
    if not metrics:
        raise ValueError(
            "No expected metric columns found in best_by_gc_rate-model-sp_reg.csv"
        )

    # We'll pivot on sp_reg; primary cell value = test_auroc_mean (or first available)
    primary = metrics[0]
    piv = df.pivot_table(
        index="gc_rate", columns="sp_reg", values=primary, aggfunc="first"
    ).sort_index(ascending=False)

    # Format numbers
    piv = piv.applymap(lambda v: _format_number(v, 3) if pd.notnull(v) else "-")

    # Bring to a nice DataFrame with index as a column name
    piv.index.name = "gc_rate"
    piv = piv.reset_index()

    caption = (
        "Performance of \\texttt{Transfer} across $\\mathrm{gc\\_rate} \\in "
        "\\{1.0, 0.5, 0.2, 0.1, 0.05\\}$ with varying $\\texttt{sp\\_reg}$ "
        f"({primary.replace('_', '\\_')})."
    )
    latex = _to_latex_table(
        piv,
        caption=caption,
        label="tab:transfer-spreg",
        index=False,
    )
    _safe_save_text(latex, merged_dir / out_name)


# -----------------------------
# 3) Transfer: gc_rates × ft_head_only 성능 비교
# -----------------------------
def make_table_transfer_headonly(
    merged_dir=MERGED_DIR,
    gc_rates=(1.0, 0.5, 0.2, 0.1, 0.05),
    out_name="table_transfer_headonly.tex",
):
    merged_dir = Path(merged_dir)
    df = _load_best_csv(["gc_rate", "model", "ft_head_only"], merged_dir)
    df = df[df["model"] == "Transfer"].copy()
    df = df[df["gc_rate"].isin(gc_rates)].copy()

    metrics = _metric_cols_available(df)
    if not metrics:
        raise ValueError(
            "No expected metric columns found in best_by_gc_rate-model-ft_head_only.csv"
        )

    primary = metrics[0]
    piv = df.pivot_table(
        index="gc_rate", columns="ft_head_only", values=primary, aggfunc="first"
    ).sort_index(ascending=False)

    # Ensure boolean-like columns appear as strings
    piv.columns = [str(c) for c in piv.columns]
    piv = piv.applymap(lambda v: _format_number(v, 3) if pd.notnull(v) else "-")

    piv.index.name = "gc_rate"
    piv = piv.reset_index()

    caption = (
        "Performance of \\texttt{Transfer} across $\\mathrm{gc\\_rate} \\in "
        "\\{1.0, 0.5, 0.2, 0.1, 0.05\\}$ with varying $\\texttt{ft\\_head\\_only}$ "
        f"({primary.replace('_', '\\_')})."
    )
    latex = _to_latex_table(
        piv,
        caption=caption,
        label="tab:transfer-headonly",
        index=False,
    )
    _safe_save_text(latex, merged_dir / out_name)


# -----------------------------
# 4) Transfer: gc_rate==1.0 에서 cancer_type 별 성능 비교
# -----------------------------
def make_table_transfer_cancertype_gc1(
    merged_dir=MERGED_DIR, out_name="table_transfer_cancertype_gc1.tex"
):
    merged_dir = Path(merged_dir)
    df = _load_best_csv(["gc_rate", "model", "cancer_type"], merged_dir)
    df = df[(df["model"] == "Transfer") & (df["gc_rate"] == 1.0)].copy()

    metrics = _metric_cols_available(df)
    if not metrics:
        raise ValueError(
            "No expected metric columns found in best_by_gc_rate-model-cancer_type.csv"
        )

    # Keep cancer_type + metrics
    cols = ["cancer_type"] + metrics
    df = df[cols].copy()

    # Format
    for m in metrics:
        df[m] = df[m].map(lambda v: _format_number(v, 3))

    latex = _to_latex_table(
        df.sort_values(by=metrics[0], ascending=False).reset_index(drop=True),
        caption=(
            "Performance of \\texttt{Transfer} at $\\mathrm{gc\\_rate}=1.0$ across different "
            "\\texttt{cancer\\_type}."
        ),
        label="tab:transfer-cancertype-gc1",
        index=False,
    )
    _safe_save_text(latex, merged_dir / out_name)


# -----------------------------
# Entrypoint (run all)
# -----------------------------
def make_all_latex_tables(merged_dir=MERGED_DIR):
    make_table_model_compare_gc1(merged_dir)
    make_table_transfer_spreg(merged_dir)
    make_table_transfer_headonly(merged_dir)
    make_table_transfer_cancertype_gc1(merged_dir)


def load_filtered_dataframe(data_dir: Path):
    """
    Load all JSONs once, filter by meta_config and HP space, and return a DataFrame.
    This isolates the expensive IO/validation step so it can be reused.
    """
    if stop_requested():
        return None

    merged_dir = Path(MERGED_DIR)
    os.makedirs(merged_dir, exist_ok=True)

    json_files = list(data_dir.glob("*.json"))
    print(f"📂 {data_dir} JSON files found: {len(json_files)}")

    # 1) Load + meta_config filter
    data_list = []
    for f in my_tqdm(json_files, desc="Filtering with meta config (single pass)"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if dic_match(data, meta_config):
                data_list.append(data)
        except Exception as e:
            print(f"⚠ {f.name} read error: {e}")
    if not data_list:
        print("⚠ No JSON collected after meta_config filtering.")
        return None

    # 2) Validate against HP space (must match exactly one)
    c_m = ConfigManager(data_list)
    filtered_data_list = []
    for setting in my_tqdm(
        safe_my_param_grid_with_modification(HP_SPACE_LIST, meta_config),
        desc="Filtering with hyperparameter space (single pass)",
    ):
        matched = c_m.find(setting)
        count = len(matched)
        assert (
            count == 1
        ), f"count={count} for setting={setting} - exp_name list: {[d['exp_name'] for d in matched]}"
        filtered_data_list.append(matched[0])

    df = pd.DataFrame(filtered_data_list)
    return df


def make_merged_results(
    data_dir: Path,
    extra_group_keys=[
        "cancer_type",
        "sp_reg",
        "mb_rate",
        "ft_head_only",
        "reg_mode",
        "lora",
    ],
    preloaded_df: pd.DataFrame | None = None,
    drop_const_cols=False,
):

    merged_dir = Path(MERGED_DIR)

    if preloaded_df is not None:
        df = preloaded_df.copy()
    else:
        df = load_filtered_dataframe(data_dir)

    # (1) 개별 결과 저장(기존 동작 유지)
    sort_cols = [c for c in ["val_auroc", "test_auroc"] if c in df.columns]
    df_sorted = df.sort_values(by=sort_cols, ascending=False) if sort_cols else df
    raw_csv_path = merged_dir / f"merged_results.csv"
    raw_xlsx_path = merged_dir / f"merged_results.xlsx"

    df_sorted.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    df_sorted.to_excel(raw_xlsx_path, index=False)
    print(f"CSV 저장 완료 → {raw_csv_path}")
    print(f"Excel 저장 완료 → {raw_xlsx_path}")

    # (2) 평균 집계
    metric_cols = get_metric_cols(df.columns)
    print(f"metric_cols: {metric_cols}")
    assert len(metric_cols) >= 1

    exclude_cols = set(
        metric_cols
        + ["exp_name", "seed", "timestamp", "best_epoch", "cv_id", "split_seed"]
    )
    group_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"group_cols: {group_cols}")

    # --- 핵심 수정: 그룹 키를 hashable로 변환 ---
    def _freeze(x):
        # 리스트/튜플/ndarray → 튜플
        import numpy as _np

        if isinstance(x, (list, tuple, _np.ndarray)):
            return tuple(_freeze(v) for v in list(x))
        # 딕셔너리 → (key, frozen(value))를 key로 정렬한 튜플
        if isinstance(x, dict):
            return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
        # 집합 → 정렬된 튜플
        if isinstance(x, set):
            return tuple(sorted(_freeze(v) for v in x))
        return x  # 숫자/문자/None 등은 그대로

    df_h = df.copy()
    for c in group_cols:
        # object 컬럼만 변환 시도 (숫자/범주형은 영향 없음)
        if df_h[c].dtype == "object":
            df_h[c] = df_h[c].map(_freeze)

    g = df_h.groupby(group_cols, dropna=False)

    agg_mean = g[metric_cols].mean().rename(columns=lambda x: f"{x}_mean")
    agg_std = g[metric_cols].std(ddof=0).rename(columns=lambda x: f"{x}_std")
    n_runs = (
        g["exp_name"].nunique().rename("n_runs")
        if "exp_name" in df_h.columns
        else g.size().rename("n_runs")
    )

    # 그룹별 exp_name 값을 tuple로 수집
    exp_names_grp = g["exp_name"].apply(lambda s: tuple(s.tolist())).rename("exp_name")

    summary = pd.concat(
        [agg_mean, agg_std, n_runs, exp_names_grp], axis=1
    ).reset_index()

    sort_key = (
        "val_auroc_mean"
        if "val_auroc_mean" in summary.columns
        else f"{metric_cols[0]}_mean"
    )
    summary = summary.sort_values(by=sort_key, ascending=False)
    if drop_const_cols:
        summary, dropped_col_dict = drop_constant_columns(summary)
        json_save_path = merged_dir / f"avg_by_seed_dropped_cols.json"
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(dropped_col_dict, f, indent=4, ensure_ascii=False)
        print(
            f"avg_by_seed에서 제거된 상수형 컬럼 정보 JSON 저장 완료 → {json_save_path}"
        )

    avg_csv_path = merged_dir / f"avg_by_seed.csv"
    avg_xlsx_path = merged_dir / f"avg_by_seed.xlsx"
    summary.to_csv(avg_csv_path, index=False, encoding="utf-8-sig")
    summary.to_excel(avg_xlsx_path, index=False)

    print(f"Seed-평균 요약 CSV 저장 완료 → {avg_csv_path}")
    print(f"Seed-평균 요약 Excel 저장 완료 → {avg_xlsx_path}")

    # (3) Select best rows per group by validation AUROC (configurable group keys)
    # ---------------------------------------------------------------------------
    # We support additional grouping columns beyond ("gc_rate", "model").
    # Users can provide extra group columns via environment variable:
    #   BEST_GROUP_EXTRA_COLS="cv_id,cancer_type,pos_weight"
    # Only columns that exist in 'summary' will be used.
    print(
        "[INFO] Selecting best rows per group by val_auroc_mean (with optional extra grouping keys)..."
    )

    if "val_auroc_mean" not in summary.columns:
        # Fallback to first available metric mean if val_auroc_mean is not present.
        # This keeps the behavior robust across different metric sets.
        metric_mean_fallback = [c for c in summary.columns if c.endswith("_mean")]
        if not metric_mean_fallback:
            print(
                "[WARN] No '*_mean' columns found. Skipping best-per-group selection."
            )
            return
        best_metric_col = (
            "val_auroc_mean"
            if "val_auroc_mean" in summary.columns
            else metric_mean_fallback[0]
        )
    else:
        best_metric_col = "val_auroc_mean"

    # Base grouping keys
    base_group_keys = ["gc_rate", "model"]
    # # Extra grouping keys from environment variable (comma-separated)
    # extra_keys_env = os.environ.get("BEST_GROUP_EXTRA_COLS", "")
    # extra_group_keys = [k.strip() for k in extra_keys_env.split(",") if k.strip()]
    # Keep keys that actually exist in the DataFrame
    exist_keys = [
        k for k in (base_group_keys + extra_group_keys) if k in summary.columns
    ]

    if not exist_keys:
        print(
            "[WARN] No valid grouping keys found among base+extra. Skipping best-per-group selection."
        )
        return

    # Sort so that the top row per group has the best validation AUROC (or fallback metric)
    sort_cols = exist_keys + [best_metric_col]
    sort_asc = [True] * len(exist_keys) + [False]
    _tmp = summary.sort_values(sort_cols, ascending=sort_asc)
    best_by_groups = _tmp.drop_duplicates(subset=exist_keys, keep="first").copy()

    # ---- Column ordering: identifiers → performance means → chosen hyperparameters (to the RIGHT)
    id_cols = [c for c in (exist_keys + ["exp_name"]) if c in best_by_groups.columns]
    metric_mean_cols = [
        c
        for c in best_by_groups.columns
        if (
            c.endswith("_mean")
            and (c.startswith("val_") or c.startswith("test_") or "ft_elapsed_sec" in c)
        )
    ]
    # Non-metric / non-std candidates (possible hyperparameters)
    non_metric_cols = [
        c
        for c in best_by_groups.columns
        if not (c.endswith("_mean") or c.endswith("_std"))
    ]
    # Exclude identifier/utility columns from hyperparam list
    exclude_for_hp = set(
        id_cols
        + [
            "n_seeds",
            "seed",
            "cv_id",
            "timestamp",
            "file_name",
            "file",
            "folds",
            "split_seed",
        ]
    )
    hyperparam_cols = [c for c in non_metric_cols if c not in exclude_for_hp]

    # Final order: ids | val/test means | hyperparameters (so HPs appear on the RIGHT)
    final_cols = [*id_cols, *metric_mean_cols, *hyperparam_cols]
    final_cols = [c for c in final_cols if c in best_by_groups.columns]  # safety
    best_by_groups = best_by_groups[final_cols].reset_index(drop=True)

    # Save to CSV/XLSX with a descriptive suffix
    suffix = "-".join(exist_keys).replace(os.sep, "_")
    best_csv_path = merged_dir / f"best_by_{suffix}.csv"
    best_xlsx_path = merged_dir / f"best_by_{suffix}.xlsx"
    best_by_groups.to_csv(best_csv_path, index=False, encoding="utf-8-sig")
    best_by_groups.to_excel(best_xlsx_path, index=False)
    print(f"Best-per-group CSV saved → {best_csv_path}")
    print(f"Best-per-group Excel saved → {best_xlsx_path}")
