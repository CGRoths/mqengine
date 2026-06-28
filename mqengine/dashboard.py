from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from flask import Flask, jsonify, render_template, request

from .result import BacktestResult, SweepResult, downsample_xy

FILTER_STATE_MAX_CURVES = 50
EQUITY_MAX_POINTS = 1200
DETAIL_CHART_MAX_POINTS = 1800


def _sort_filtered_df(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if df.empty:
        return df
    if sort_by in df.columns and pd.api.types.is_numeric_dtype(df[sort_by]):
        return df.sort_values([sort_by, "return_pct"], ascending=[False, False]).reset_index(drop=True)
    if sort_by == "sharpe":
        return df.sort_values(["sharpe", "return_pct"], ascending=[False, False]).reset_index(drop=True)
    if sort_by == "calmar":
        return df.sort_values(["calmar", "return_pct"], ascending=[False, False]).reset_index(drop=True)
    if sort_by == "return_pct":
        return df.sort_values(["return_pct", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    if sort_by == "max_drawdown_asc":
        return df.sort_values(["max_drawdown", "sharpe"], ascending=[True, False]).reset_index(drop=True)
    if sort_by == "num_trades":
        return df.sort_values(["num_trades", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    return df.sort_values(["sharpe", "return_pct"], ascending=[False, False]).reset_index(drop=True)


def _apply_filters(df: pd.DataFrame, filters: dict[str, list[Any]]) -> pd.DataFrame:
    out = df.copy()
    for key, vals in (filters or {}).items():
        if key not in out.columns or not vals:
            continue
        out = out[out[key].astype(str).isin([str(v) for v in vals])]
    return out


def _build_filter_options(sweep_result: SweepResult) -> dict[str, list[Any]]:
    return sweep_result.filter_options()


def _build_metric_payload(results_df: pd.DataFrame, metric_col: str, title: str, description: str) -> dict:
    if metric_col not in results_df.columns:
        return {"description": f"{title} is not available for this result set.", "chart": {"traces": [], "layout": {"title": title}}}

    if "rolling_window" in results_df.columns and "short_threshold" in results_df.columns:
        heatmap_df = results_df.groupby(["rolling_window", "short_threshold"], as_index=False)[metric_col].max()
        x_vals = sorted(heatmap_df["short_threshold"].unique().tolist())
        y_vals = sorted(heatmap_df["rolling_window"].unique().tolist())
        z = []
        for rw in y_vals:
            row = []
            for thr in x_vals:
                match = heatmap_df[(heatmap_df["rolling_window"] == rw) & (heatmap_df["short_threshold"] == thr)]
                row.append(float(match[metric_col].iloc[0]) if not match.empty else None)
            z.append(row)
        return {
            "description": description,
            "chart": {
                "traces": [{
                    "x": x_vals,
                    "y": y_vals,
                    "z": z,
                    "type": "heatmap",
                    "hovertemplate": f"Short Thr=%{{x}}<br>Rolling=%{{y}}<br>{title}=%{{z}}<extra></extra>",
                }],
                "layout": {"title": title, "xaxis": {"title": "Short Threshold"}, "yaxis": {"title": "Rolling Window"}},
            },
        }

    top = results_df.sort_values(metric_col, ascending=False).head(20)
    return {
        "description": description,
        "chart": {
            "traces": [{
                "x": top["strategy_id"].tolist(),
                "y": top[metric_col].astype(float).tolist(),
                "type": "bar",
                "hovertemplate": f"Strategy=%{{x}}<br>{title}=%{{y}}<extra></extra>",
            }],
            "layout": {"title": title, "xaxis": {"title": "Strategy"}, "yaxis": {"title": title}},
        },
    }


def _build_mode_payload(results_df: pd.DataFrame, mode_key: str) -> dict:
    if results_df is None or results_df.empty:
        return {"description": "No results available.", "chart": {"traces": [], "layout": {"title": "No Data"}}}

    metric_modes = {
        "trade_sharpe": ("trade_sharpe", "Trade Sharpe Heatmap", "Trade-level Sharpe computed from closed-trade account returns."),
        "period_sharpe": ("period_sharpe", "Period Sharpe Heatmap", "Period Sharpe computed from equity curve period returns."),
        "sortino": ("sortino", "Sortino Heatmap", "Downside-risk-adjusted period return."),
        "calmar": ("calmar", "Calmar Heatmap", "CAGR divided by max drawdown."),
        "max_drawdown": ("max_drawdown", "Max Drawdown Heatmap", "Maximum drawdown percentage; lower is better."),
        "stability": ("validation_is_oos_consistency_score", "Parameter Stability", "IS/OOS consistency score across parameter sets."),
    }
    if mode_key in metric_modes:
        metric_col, title, description = metric_modes[mode_key]
        return _build_metric_payload(results_df, metric_col, title, description)

    if mode_key == "is_oos":
        if "in_sample_period_sharpe" in results_df.columns and "out_sample_period_sharpe" in results_df.columns:
            top = results_df.sort_values("period_sharpe", ascending=False).head(20)
            return {
                "description": "Compare in-sample and out-of-sample period Sharpe for top full-period strategies.",
                "chart": {
                    "traces": [
                        {"x": top["strategy_id"].tolist(), "y": top["in_sample_period_sharpe"].astype(float).tolist(), "type": "bar", "name": "In Sample"},
                        {"x": top["strategy_id"].tolist(), "y": top["out_sample_period_sharpe"].astype(float).tolist(), "type": "bar", "name": "Out Sample"},
                    ],
                    "layout": {"title": "IS/OOS Period Sharpe", "barmode": "group"},
                },
            }
        return {"description": "IS/OOS metrics are available after running sweep.run_research(...).", "chart": {"traces": [], "layout": {"title": "IS/OOS Comparison"}}}

    if mode_key == "sharpe":
        if "rolling_window" in results_df.columns and "short_threshold" in results_df.columns:
            heatmap_df = results_df.groupby(["rolling_window", "short_threshold"], as_index=False)["sharpe"].max()
            x_vals = sorted(heatmap_df["short_threshold"].unique().tolist())
            y_vals = sorted(heatmap_df["rolling_window"].unique().tolist())
            z = []
            for rw in y_vals:
                row = []
                for thr in x_vals:
                    match = heatmap_df[(heatmap_df["rolling_window"] == rw) & (heatmap_df["short_threshold"] == thr)]
                    row.append(float(match["sharpe"].iloc[0]) if not match.empty else None)
                z.append(row)
            return {
                "description": "Best in-sample trade-level Sharpe across rolling_window × short_threshold.",
                "chart": {
                    "traces": [{
                        "x": x_vals,
                        "y": y_vals,
                        "z": z,
                        "type": "heatmap",
                        "hovertemplate": "Short Thr=%{x}<br>Rolling=%{y}<br>Sharpe=%{z}<extra></extra>",
                    }],
                    "layout": {
                        "title": "Trade-Level Sharpe Heatmap",
                        "xaxis": {"title": "Short Threshold"},
                        "yaxis": {"title": "Rolling Window"},
                    },
                },
            }
        top = results_df.head(20)
        return {
            "description": "Top strategies ranked by Sharpe.",
            "chart": {
                "traces": [{
                    "x": top["strategy_id"].tolist(),
                    "y": top["sharpe"].tolist(),
                    "type": "bar",
                    "hovertemplate": "Strategy=%{x}<br>Sharpe=%{y}<extra></extra>",
                }],
                "layout": {"title": "Top Sharpe Strategies", "xaxis": {"title": "Strategy"}, "yaxis": {"title": "Sharpe"}},
            },
        }
    return {"description": f"{mode_key} mode placeholder.", "chart": {"traces": [], "layout": {"title": f"{mode_key.title()} Mode"}}}


def build_single_dashboard_app(result: BacktestResult) -> Flask:
    template_dir = str(Path(__file__).resolve().parent / "templates")
    app = Flask(__name__, template_folder=template_dir)
    detail_payload = result.to_detail_payload(max_points=DETAIL_CHART_MAX_POINTS, benchmark_name=result.meta.get("benchmark_name", "BUY_AND_HOLD"))
    meta = {
        "page_title": f"MQENGINE · {result.name}",
        "dataset_name": result.name,
        "requested_start": detail_payload["usable_start"],
        "requested_end": detail_payload["usable_end"],
        "total_strategies": 1,
    }

    @app.get("/")
    def home():
        return render_template("basic.html")

    @app.get("/strategy/<strategy_id>")
    def strategy_page(strategy_id: str):
        return render_template("basic.html")

    @app.get("/api/meta")
    def api_meta():
        return jsonify(meta)

    @app.get("/api/benchmark")
    def api_benchmark():
        chart = detail_payload["chart_data"]["equity_chart"]
        return jsonify({
            "name": result.meta.get("benchmark_name", "BUY_AND_HOLD"),
            "metrics": {},
            "equity_dates": chart["benchmark_dates"],
            "equity": chart["benchmark_equity"],
            "actual_start": detail_payload["usable_start"],
            "actual_end": detail_payload["usable_end"],
        })

    @app.get("/api/filter-options")
    def api_filter_options():
        return jsonify({})

    @app.get("/api/modes")
    def api_modes():
        return jsonify({"modes": []})

    @app.get("/api/mode/<mode_key>")
    def api_mode(mode_key: str):
        return jsonify({"description": "Single result mode.", "chart": {"traces": [], "layout": {"title": "Single Result"}}})

    @app.post("/api/filter")
    def api_filter():
        x, y = downsample_xy(detail_payload["chart_data"]["equity_chart"]["strategy_dates"], detail_payload["chart_data"]["equity_chart"]["strategy_equity"], max_points=EQUITY_MAX_POINTS)
        return jsonify({
            "filtered_count": 1,
            "curve_count": 1,
            "top_strategy": {"strategy_id": result.strategy_id, "metrics": result.metrics},
            "strategies": [{
                "strategy_id": result.strategy_id,
                "name": result.name,
                "code_ref": "mqengine/single",
                "params": result.params,
                "metrics": result.metrics,
                "usable_start": detail_payload["usable_start"],
                "usable_end": detail_payload["usable_end"],
                "usable_rows": detail_payload["usable_rows"],
            }],
            "curves": [{"strategy_id": result.strategy_id, "label": result.strategy_id, "equity_dates": x, "equity": y}],
        })

    @app.get("/api/strategy/<strategy_id>")
    def api_strategy(strategy_id: str):
        if strategy_id != result.strategy_id:
            return jsonify({"error": "not_found"}), 404
        return jsonify(detail_payload)

    return app


def build_sweep_dashboard_app(sweep_result: SweepResult) -> Flask:
    template_dir = str(Path(__file__).resolve().parent / "templates")
    app = Flask(__name__, template_folder=template_dir)
    results_df = sweep_result.results_df.copy()
    details = {r.strategy_id: r for r in sweep_result.strategy_results}
    filter_options = _build_filter_options(sweep_result)
    modes = [
        {"key": "trade_sharpe", "label": "Trade Sharpe Heatmap"},
        {"key": "period_sharpe", "label": "Period Sharpe Heatmap"},
        {"key": "sortino", "label": "Sortino Heatmap"},
        {"key": "calmar", "label": "Calmar Heatmap"},
        {"key": "max_drawdown", "label": "Max Drawdown Heatmap"},
        {"key": "is_oos", "label": "IS/OOS Comparison"},
        {"key": "stability", "label": "Parameter Stability"},
        {"key": "walkforward", "label": "Walk-Forward"},
        {"key": "montecarlo", "label": "Monte Carlo"},
    ]
    first_result = sweep_result.strategy_results[0] if sweep_result.strategy_results else None

    meta = {
        "page_title": sweep_result.meta.get("page_title", f"MQENGINE · {sweep_result.name}"),
        "dataset_name": sweep_result.meta.get("dataset_name", sweep_result.name),
        "requested_start": first_result.data["ts"].iloc[0].strftime("%Y-%m-%d") if first_result else None,
        "requested_end": first_result.data["ts"].iloc[-1].strftime("%Y-%m-%d") if first_result else None,
        "total_strategies": int(len(results_df)),
        **{k: v for k, v in sweep_result.meta.items() if k not in {"page_title", "dataset_name"}},
    }

    @app.get("/")
    def home():
        return render_template("basic.html")

    @app.get("/strategy/<strategy_id>")
    def strategy_page(strategy_id: str):
        return render_template("basic.html")

    @app.get("/api/meta")
    def api_meta():
        return jsonify(meta)

    @app.get("/api/benchmark")
    def api_benchmark():
        if not first_result or not first_result.benchmark_col:
            return jsonify({"name": "BUY_AND_HOLD", "metrics": {}, "equity_dates": [], "equity": [], "actual_start": None, "actual_end": None})
        dates = first_result.data["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
        equity = first_result.data[first_result.benchmark_col].astype(float).tolist()
        x, y = downsample_xy(dates, equity, max_points=EQUITY_MAX_POINTS)
        return jsonify({
            "name": first_result.meta.get("benchmark_name", "BUY_AND_HOLD"),
            "metrics": {},
            "equity_dates": x,
            "equity": y,
            "actual_start": first_result.data["ts"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "actual_end": first_result.data["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        })

    @app.get("/api/filter-options")
    def api_filter_options():
        return jsonify(filter_options)

    @app.get("/api/modes")
    def api_modes():
        return jsonify({"modes": modes})

    @app.get("/api/mode/<mode_key>")
    def api_mode(mode_key: str):
        return jsonify(_build_mode_payload(results_df, mode_key))

    @app.post("/api/filter")
    def api_filter():
        payload = request.get_json(silent=True) or {}
        filters = payload.get("filters", {})
        sort_by = payload.get("sort_by", "sharpe")
        top_n = int(payload.get("top_n", FILTER_STATE_MAX_CURVES))
        filtered = _apply_filters(results_df, filters)
        filtered = _sort_filtered_df(filtered, sort_by)

        strategies = []
        for _, row in filtered.iterrows():
            params = {k: row[k] for k in sweep_result.param_columns if k in row}
            metric_keys = [
                "return_pct",
                "sharpe",
                "trade_sharpe",
                "period_sharpe",
                "sortino",
                "max_drawdown",
                "net_profit",
                "cagr_pct",
                "calmar",
                "num_trades",
                "win_rate",
                "profit_factor",
                "avg_trade_pct",
                "trades_per_year",
            ]
            metrics = {}
            for key in metric_keys:
                if key not in row or pd.isna(row[key]):
                    continue
                metrics[key] = int(row[key]) if key == "num_trades" else float(row[key])
            strategies.append({
                "strategy_id": row["strategy_id"],
                "name": row["name"],
                "code_ref": "mqengine/sweep",
                "params": params,
                "metrics": metrics,
                "usable_start": row["usable_start"],
                "usable_end": row["usable_end"],
                "usable_rows": int(row["usable_rows"]),
            })

        curves = []
        for _, row in filtered.head(top_n).iterrows():
            detail = details[row["strategy_id"]]
            x = detail.data["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
            y = detail.data["equity"].astype(float).tolist()
            x, y = downsample_xy(x, y, max_points=EQUITY_MAX_POINTS)
            curves.append({
                "strategy_id": detail.strategy_id,
                "label": detail.strategy_id,
                "equity_dates": x,
                "equity": y,
            })

        top_strategy = strategies[0] if strategies else None
        return jsonify({
            "filtered_count": int(len(filtered)),
            "curve_count": int(len(curves)),
            "strategies": strategies,
            "curves": curves,
            "top_strategy": top_strategy,
        })

    @app.get("/api/strategy/<strategy_id>")
    def api_strategy(strategy_id: str):
        if strategy_id not in details:
            return jsonify({"error": "not_found"}), 404
        result = details[strategy_id]
        payload = result.to_detail_payload(max_points=DETAIL_CHART_MAX_POINTS, benchmark_name=result.meta.get("benchmark_name", "BUY_AND_HOLD"))
        payload["benchmark"]["metrics"] = {}
        return jsonify(payload)

    return app
