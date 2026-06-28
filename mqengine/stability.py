from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numeric_param_columns(df: pd.DataFrame, param_columns: list[str]) -> list[str]:
    out = []
    for col in param_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    valid = pd.concat([a, b], axis=1).dropna()
    if len(valid) < 2:
        return 0.0
    corr = valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank())
    if pd.isna(corr):
        return 0.0
    return float(corr)


def compute_parameter_stability(
    results_df: pd.DataFrame,
    param_columns: list[str],
    *,
    objective: str = "period_sharpe",
) -> dict[str, Any]:
    if results_df is None or results_df.empty:
        return {
            "parameter_stability_score": 0.0,
            "neighbor_sharpe_decay": 0.0,
            "top_decile_consistency": 0.0,
            "heatmap_smoothness_score": 0.0,
            "oos_rank_consistency": 0.0,
            "best_is_params": {},
            "best_oos_params": {},
            "is_oos_rank_corr": 0.0,
        }

    df = results_df.copy()
    metric_col = objective if objective in df.columns else ("sharpe" if "sharpe" in df.columns else df.select_dtypes("number").columns[-1])
    numeric_params = _numeric_param_columns(df, param_columns)
    best_idx = df[metric_col].astype(float).idxmax()
    best = df.loc[best_idx]
    best_metric = float(best[metric_col])

    neighbor_decay = 0.0
    smoothness = 0.0
    if numeric_params and len(df) > 1:
        distances = []
        for col in numeric_params:
            span = max(float(df[col].max() - df[col].min()), 1e-12)
            distances.append(((df[col].astype(float) - float(best[col])) / span) ** 2)
        distance = np.sqrt(np.sum(distances, axis=0))
        positive = distance[distance > 0.0]
        if len(positive):
            threshold = np.nanquantile(positive, 0.25)
            neighbors = df[(distance > 0.0) & (distance <= threshold)]
            if not neighbors.empty and abs(best_metric) > 1e-12:
                neighbor_decay = float((best_metric - neighbors[metric_col].astype(float).mean()) / abs(best_metric))
        ordered = df.sort_values(numeric_params)
        diffs = ordered[metric_col].astype(float).diff().dropna().abs()
        smoothness = 1.0 / (1.0 + float(diffs.mean())) if not diffs.empty else 1.0

    cutoff = max(int(np.ceil(len(df) * 0.10)), 1)
    top = df.sort_values(metric_col, ascending=False).head(cutoff)
    top_decile_consistency = float((top[metric_col].astype(float) > 0.0).mean()) if not top.empty else 0.0

    is_col = f"in_sample_{metric_col}"
    oos_col = f"out_sample_{metric_col}"
    if is_col not in df.columns:
        is_col = "in_sample_period_sharpe" if "in_sample_period_sharpe" in df.columns else ""
    if oos_col not in df.columns:
        oos_col = "out_sample_period_sharpe" if "out_sample_period_sharpe" in df.columns else ""

    if is_col and oos_col:
        rank_corr = _rank_corr(df[is_col].astype(float), df[oos_col].astype(float))
        best_is = df.sort_values(is_col, ascending=False).iloc[0]
        best_oos = df.sort_values(oos_col, ascending=False).iloc[0]
        best_is_params = {col: best_is[col] for col in param_columns if col in best_is}
        best_oos_params = {col: best_oos[col] for col in param_columns if col in best_oos}
        oos_rank_consistency = max(rank_corr, 0.0)
    else:
        rank_corr = 0.0
        best_is_params = {col: best[col] for col in param_columns if col in best}
        best_oos_params = {}
        oos_rank_consistency = 0.0

    score = 100.0
    score -= max(neighbor_decay, 0.0) * 35.0
    score += top_decile_consistency * 15.0
    score += smoothness * 20.0
    score += oos_rank_consistency * 30.0
    score = max(min(score, 100.0), 0.0)

    return {
        "parameter_stability_score": round(float(score), 3),
        "neighbor_sharpe_decay": round(float(neighbor_decay), 6),
        "top_decile_consistency": round(float(top_decile_consistency), 6),
        "heatmap_smoothness_score": round(float(smoothness), 6),
        "oos_rank_consistency": round(float(oos_rank_consistency), 6),
        "best_is_params": best_is_params,
        "best_oos_params": best_oos_params,
        "is_oos_rank_corr": round(float(rank_corr), 6),
    }
