#!/usr/bin/env python3
"""Generate high-conviction next-day ETF signals for the top 100 US-listed ETFs.

The pipeline:
 1. Loads a curated universe of non-leveraged ETFs.
 2. Downloads ~24 months of daily prices plus macro/market proxies.
 3. Engineers momentum, trend, volume, cross-sectional rank and macro features.
 4. Trains a Decision-Tree next-day bucket classifier with a time-series split.
 5. Calibrates high-conviction leaves on a validation set so the subset of
    emitted signals has >80% historical accuracy.
 6. Evaluates the high-conviction subset on an untouched test set.
 7. Emits machine-readable JSON + HTML screener for the most recent trading day.

Outputs:
    etf_signals/output/latest_signals.json
    etf_signals/output/latest_signals.html

No explicit buy/sell recommendations are produced; only directional
probability forecasts and high-conviction historical accuracy stats.
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = ROOT / "data" / "top100_etf_universe.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

LEVERAGED_KEYWORDS = [
    "leveraged",
    "2x",
    "3x",
    "ultra",
    "inverse",
    "short",
    "ultrapro",
    "bear",
    "bull",
]

BUCKET_LABELS = ["Major Down", "Mildly Down", "Mildly Up", "Major Up"]

# Market/macro proxies used for relative-strength and regime features.
MACRO_TICKERS = ["SPY", "^VIX", "UUP", "GLD", "TLT", "HYG", "LQD", "USO", "CPER"]

# Leaf purity parameters for high-conviction selection.
MIN_LEAF_SAMPLES = 20
MIN_LEAF_ACCURACY = 0.80


def is_non_leveraged(name: str) -> bool:
    lower = name.lower()
    return not any(kw in lower for kw in LEVERAGED_KEYWORDS)


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_PATH)
    df["non_leveraged"] = df["name"].apply(is_non_leveraged)
    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fetch_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """Download adjusted daily close/volume for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    return data


def build_panel(data: pd.DataFrame, tickers: list[str], names: dict[str, str]) -> pd.DataFrame:
    """Build a flat panel DataFrame from a yfinance multi-ticker download."""
    panels: list[pd.DataFrame] = []
    multi = isinstance(data.columns, pd.MultiIndex)
    for ticker in tickers:
        try:
            if multi:
                df = data[ticker].copy()
            else:
                df = data.copy()
            df = df.reset_index().rename(
                columns={"Date": "date", "Close": "close", "Volume": "volume"}
            )
            if df.empty or "close" not in df.columns or df["close"].isna().all():
                continue
            df["Ticker"] = ticker
            df["name"] = names.get(ticker, ticker)
            panels.append(df[["date", "Ticker", "name", "close", "volume"]])
        except Exception:
            continue
    return pd.concat(panels, ignore_index=True)


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker momentum, trend, volume and RSI features."""
    panel = panel.sort_values(["Ticker", "date"]).copy()
    panel["log_ret"] = panel.groupby("Ticker")["close"].transform(lambda s: np.log(s / s.shift(1)))

    for horizon in [1, 5, 10, 20, 60]:
        panel[f"return_{horizon}d_pct"] = panel.groupby("Ticker")["close"].transform(
            lambda s: (s / s.shift(horizon) - 1) * 100
        )

    panel["ma20"] = panel.groupby("Ticker")["close"].transform(lambda s: s.rolling(20).mean())
    panel["ma50"] = panel.groupby("Ticker")["close"].transform(lambda s: s.rolling(50).mean())
    panel["distance_20ma_pct"] = (panel["close"] / panel["ma20"] - 1) * 100
    panel["distance_50ma_pct"] = (panel["close"] / panel["ma50"] - 1) * 100
    panel["rsi_14"] = panel.groupby("Ticker")["close"].transform(lambda s: rsi(s, 14))
    panel["volatility_20d_annualized_pct"] = panel.groupby("Ticker")["log_ret"].transform(
        lambda s: s.rolling(20).std() * np.sqrt(252) * 100
    )
    panel["volume_ratio_20d_60d"] = panel.groupby("Ticker")["volume"].transform(
        lambda s: s.rolling(20).mean() / s.rolling(60).mean()
    )

    panel["next_day_return_pct"] = panel.groupby("Ticker")["close"].transform(
        lambda s: (s.shift(-1) / s - 1) * 100
    )
    return panel


def add_macro_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Fetch macro proxies and merge return/volatility regime features."""
    print("Downloading macro/market proxies...")
    macro_data = fetch_prices(MACRO_TICKERS, period="2y")
    if macro_data.empty:
        return panel

    macro_names = {t: t for t in MACRO_TICKERS}
    macro_panel = build_panel(macro_data, MACRO_TICKERS, macro_names)
    macro_panel = macro_panel.sort_values(["Ticker", "date"])

    wide = macro_panel.pivot(index="date", columns="Ticker", values="close").copy()
    if wide.empty:
        return panel

    # Strip carets from index tickers so derived column names are plain.
    wide = wide.rename(columns=lambda c: c.replace("^", ""))

    for t in wide.columns:
        for horizon in [1, 5, 20]:
            wide[f"{t}_ret_{horizon}d"] = wide[t].pct_change(horizon) * 100

    if "VIX" in wide.columns:
        wide["vix_level"] = wide["VIX"]
        wide["vix_1d_chg"] = wide["VIX"].diff()

    if "HYG" in wide.columns and "LQD" in wide.columns:
        wide["hyg_lqd_ratio_5d"] = (wide["HYG"] / wide["LQD"]).pct_change(5) * 100

    keep = [c for c in wide.columns if c not in [t.replace("^", "") for t in MACRO_TICKERS]]
    wide = wide[keep].ffill().reset_index()

    return panel.merge(wide, on="date", how="left", suffixes=("", ""))


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional percentile ranks and relative-vs-SPY returns."""
    rank_cols = ["return_1d_pct", "return_5d_pct", "rsi_14", "volatility_20d_annualized_pct"]
    for col in rank_cols:
        if col in panel.columns:
            panel[f"rank_{col}"] = panel.groupby("date")[col].rank(pct=True)

    if "SPY_ret_1d" in panel.columns:
        panel["rel_1d"] = panel["return_1d_pct"] - panel["SPY_ret_1d"]
    if "SPY_ret_5d" in panel.columns:
        panel["rel_5d"] = panel["return_5d_pct"] - panel["SPY_ret_5d"]
    return panel


def bucket_boundaries(series: pd.Series) -> tuple[float, float, float]:
    clean = series.dropna()
    q25, q50, q75 = clean.quantile([0.25, 0.50, 0.75])
    return float(q25), float(q50), float(q75)


def bucket_returns(series: pd.Series, q25: float, q50: float, q75: float) -> pd.Series:
    labels = np.select(
        [series <= q25, series <= q50, series <= q75, series > q75],
        [0, 1, 2, 3],
        default=2,
    )
    return pd.Series(labels, index=series.index).astype(int)


def get_feature_columns(panel: pd.DataFrame) -> tuple[list[str], list[str]]:
    base = [
        "return_1d_pct",
        "return_5d_pct",
        "return_10d_pct",
        "return_20d_pct",
        "return_60d_pct",
        "distance_20ma_pct",
        "distance_50ma_pct",
        "rsi_14",
        "volatility_20d_annualized_pct",
        "volume_ratio_20d_60d",
    ]
    extras = [c for c in panel.columns if c.startswith("rank_") or c.startswith("rel_")]
    macro = [c for c in panel.columns if "_ret_" in c or c in ("vix_level", "vix_1d_chg", "hyg_lqd_ratio_5d")]
    numeric = [c for c in base + extras + macro if c in panel.columns]
    categorical = ["Ticker"]
    return numeric, categorical


def build_model(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Train a DecisionTree next-day bucket classifier."""
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ]
    )

    clf = DecisionTreeClassifier(
        max_leaf_nodes=80,
        min_samples_leaf=200,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipeline.fit(df[numeric + categorical], df["bucket"])
    return pipeline


def calibrate_leaves(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    min_samples: int = MIN_LEAF_SAMPLES,
    min_accuracy: float = MIN_LEAF_ACCURACY,
) -> dict[int, tuple[int, float, int]]:
    """Identify decision-tree leaves that are historically pure on validation."""
    leaves = model.named_steps["clf"].apply(model.named_steps["prep"].transform(X))
    leaf_map: dict[int, tuple[int, float, int]] = {}
    for leaf in np.unique(leaves):
        mask = leaves == leaf
        if mask.sum() < min_samples:
            continue
        majority = int(pd.Series(y[mask]).mode()[0])
        acc = float((y[mask] == majority).mean())
        if acc >= min_accuracy:
            leaf_map[int(leaf)] = (majority, acc, int(mask.sum()))
    return leaf_map


def high_conviction_leaf_predictions(
    model: Pipeline,
    X: pd.DataFrame,
    leaf_map: dict[int, tuple[int, float, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_class, leaf_historical_accuracy) arrays; -1/NaN when no leaf fires."""
    leaves = model.named_steps["clf"].apply(model.named_steps["prep"].transform(X))
    preds = np.full(len(leaves), -1, dtype=int)
    hist_acc = np.full(len(leaves), np.nan, dtype=float)
    for i, leaf in enumerate(leaves):
        if leaf in leaf_map:
            preds[i] = leaf_map[leaf][0]
            hist_acc[i] = leaf_map[leaf][1]
    return preds, hist_acc


def train_and_calibrate(
    panel: pd.DataFrame,
    live_df: pd.DataFrame,
) -> tuple[Pipeline, dict[int, tuple[int, float, int]], dict[str, Any], list[dict]]:
    """Train, calibrate high-conviction leaves, backtest, and predict live."""
    numeric, categorical = get_feature_columns(panel)

    # Time-series split: train 60%, validation 20%, test 20% of historical dates.
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train = panel[panel["date"].isin(train_dates)].copy()
    val = panel[panel["date"].isin(val_dates)].copy()
    test = panel[panel["date"].isin(test_dates)].copy()

    if len(train) < 100 or len(val) < 50 or len(test) < 50:
        raise RuntimeError("Insufficient historical data for train/val/test split.")

    print(f"Training on {len(train)} rows, validating on {len(val)}, testing on {len(test)}...")

    # 1. Train on the early training set.
    final_model = build_model(train, numeric, categorical)

    # 2. Calibrate high-conviction leaves on the validation set.
    leaf_map = calibrate_leaves(final_model, val[numeric + categorical], val["bucket"].values)

    # 3. Evaluate high-conviction subset on the untouched test set.
    test_preds, _ = high_conviction_leaf_predictions(
        final_model, test[numeric + categorical], leaf_map
    )
    test_y = test["bucket"].values
    mask = test_preds != -1
    backtest_accuracy = float((test_y[mask] == test_preds[mask]).mean()) if mask.any() else 0.0
    backtest_signals = int(mask.sum())

    bucket_means = panel.groupby("bucket")["next_day_return_pct"].mean().to_dict()

    # 4. Live prediction for the most recent day.
    live_X = live_df[numeric + categorical]
    live_probs = final_model.predict_proba(live_X)
    live_preds = live_probs.argmax(axis=1)
    hc_preds, hc_acc = high_conviction_leaf_predictions(final_model, live_X, leaf_map)

    live_records: list[dict] = []
    for i, (_, row) in enumerate(live_df.iterrows()):
        pred_bucket = int(live_preds[i])
        hc_bucket = int(hc_preds[i])
        high_conviction = hc_bucket != -1
        if high_conviction:
            pred_bucket = hc_bucket
            confidence = float(live_probs[i, pred_bucket])
            historical_accuracy = float(hc_acc[i])
        else:
            confidence = float(live_probs[i, pred_bucket])
            historical_accuracy = None

        expected_return = sum(
            live_probs[i, b] * bucket_means.get(b, 0.0) for b in range(4)
        )

        record = {
            "symbol": str(row["Ticker"]),
            "name": str(row["name"]),
            "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
            "next_day_prediction": BUCKET_LABELS[pred_bucket],
            "confidence": round(confidence, 6),
            "high_conviction": high_conviction,
            "historical_accuracy": round(historical_accuracy, 4) if historical_accuracy is not None else None,
            "expected_next_day_return_pct": round(expected_return, 4),
            "prob_Major_Down": round(float(live_probs[i, 0]), 6),
            "prob_Mildly_Down": round(float(live_probs[i, 1]), 6),
            "prob_Mildly_Up": round(float(live_probs[i, 2]), 6),
            "prob_Major_Up": round(float(live_probs[i, 3]), 6),
            "return_1d_pct": round(float(row["return_1d_pct"]), 4) if pd.notna(row["return_1d_pct"]) else None,
            "return_5d_pct": round(float(row["return_5d_pct"]), 4) if pd.notna(row["return_5d_pct"]) else None,
            "return_10d_pct": round(float(row["return_10d_pct"]), 4) if pd.notna(row["return_10d_pct"]) else None,
            "return_20d_pct": round(float(row["return_20d_pct"]), 4) if pd.notna(row["return_20d_pct"]) else None,
            "return_60d_pct": round(float(row["return_60d_pct"]), 4) if pd.notna(row["return_60d_pct"]) else None,
            "rsi_14": round(float(row["rsi_14"]), 2) if pd.notna(row["rsi_14"]) else None,
            "distance_20ma_pct": round(float(row["distance_20ma_pct"]), 4) if pd.notna(row["distance_20ma_pct"]) else None,
            "distance_50ma_pct": round(float(row["distance_50ma_pct"]), 4) if pd.notna(row["distance_50ma_pct"]) else None,
            "volume_ratio_20d_60d": round(float(row["volume_ratio_20d_60d"]), 4) if pd.notna(row["volume_ratio_20d_60d"]) else None,
        }
        live_records.append(record)

    # High-conviction first, then by confidence.
    live_records.sort(key=lambda r: (not r["high_conviction"], -r["confidence"]))
    for idx, r in enumerate(live_records, start=1):
        r["rank"] = idx

    leaf_summary = {
        f"leaf_{leaf}": {
            "predicted_class": BUCKET_LABELS[pred_class],
            "validation_accuracy": round(acc, 4),
            "validation_samples": n,
        }
        for leaf, (pred_class, acc, n) in leaf_map.items()
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backtest_period_start": str(test_dates[0]) if len(test_dates) else None,
        "backtest_period_end": str(test_dates[-1]) if len(test_dates) else None,
        "backtest_high_conviction_accuracy": round(backtest_accuracy, 6),
        "backtest_high_conviction_signals": backtest_signals,
        "high_conviction_leaves": leaf_summary,
        "bucket_mean_next_day_return_pct": {BUCKET_LABELS[k]: round(float(v), 6) for k, v in bucket_means.items()},
    }

    return final_model, leaf_map, summary, live_records


def build_html(records: list[dict], summary: dict) -> str:
    hc_count = sum(1 for r in records if r["high_conviction"])
    rows = ""
    for r in records:
        if "Up" in r["next_day_prediction"]:
            cls = "bull"
        elif "Down" in r["next_day_prediction"]:
            cls = "bear"
        else:
            cls = "neutral"
        badge = "<span class='badge'>HIGH CONVICTION</span>" if r["high_conviction"] else ""
        rows += (
            f"<tr class='{cls}'>"
            f"<td>{r['rank']}</td>"
            f"<td><b>{r['symbol']}</b><br><small>{r['name']}</small></td>"
            f"<td>{r['next_day_prediction']} {badge}</td>"
            f"<td>{r['confidence']:.1%}</td>"
            f"<td>{(r['historical_accuracy'] or 0):.1%}</td>"
            f"<td>{r['expected_next_day_return_pct']:.2f}%</td>"
            f"<td>{r['prob_Major_Up']:.1%}</td>"
            f"<td>{r['prob_Mildly_Up']:.1%}</td>"
            f"<td>{r['prob_Mildly_Down']:.1%}</td>"
            f"<td>{r['prob_Major_Down']:.1%}</td>"
            f"<td>{r['return_1d_pct']:.2f}%</td>"
            f"<td>{r['return_5d_pct']:.2f}%</td>"
            f"<td>{r['return_20d_pct']:.2f}%</td>"
            f"<td>{r['rsi_14']:.1f}</td>"
            f"</tr>\n"
        )

    backtest_acc = summary["backtest_high_conviction_accuracy"]
    backtest_sigs = summary["backtest_high_conviction_signals"]

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Top 100 ETF Next-Day Screener</title>
<style>
:root {{ color-scheme: dark; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background:#0d1117; color:#c9d1d9; margin:0; padding:20px; }}
.container {{ max-width:1400px; margin:0 auto; }}
h2 {{ color:#58a6ff; margin-bottom:6px; }}
.meta {{ color:#8b949e; margin-bottom:20px; font-size:14px; }}
.summary {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; min-width:180px; }}
.card .label {{ color:#8b949e; font-size:12px; text-transform:uppercase; }}
.card .value {{ color:#58a6ff; font-size:24px; font-weight:600; }}
table {{ border-collapse:collapse; width:100%; margin-top:10px; font-size:13px; }}
th {{ background:#21262d; color:#58a6ff; padding:10px; text-align:left; border-bottom:2px solid #30363d; position:sticky; top:0; }}
td {{ padding:8px 10px; border-bottom:1px solid #30363d; vertical-align:top; }}
tr:hover {{ background:#161b22; }}
.bull {{ color:#3fb950; }}
.bear {{ color:#f85149; }}
.neutral {{ color:#8b949e; }}
.badge {{ background:#238636; color:#fff; font-size:10px; padding:2px 6px; border-radius:4px; margin-left:6px; }}
.disclaimer {{ color:#8b949e; font-size:12px; margin-top:20px; line-height:1.5; }}
</style>
</head>
<body>
<div class="container">
<h2>Top 100 US-Listed Non-Leveraged ETF Next-Day Screener</h2>
<p class="meta">Generated at {summary['generated_at']} UTC | Out-of-sample backtest {summary.get('backtest_period_start','')} to {summary.get('backtest_period_end','')}: <b>{backtest_acc:.1%}</b> high-conviction accuracy on {backtest_sigs} signals</p>
<div class="summary">
  <div class="card"><div class="label">High-Conviction Signals</div><div class="value">{hc_count}</div></div>
  <div class="card"><div class="label">Backtest Accuracy</div><div class="value">{backtest_acc:.1%}</div></div>
  <div class="card"><div class="label">Backtest Signals</div><div class="value">{backtest_sigs}</div></div>
</div>
<table>
<tr>
<th>Rank</th><th>Ticker</th><th>Next-Day</th><th>Confidence</th><th>Hist. Acc.</th><th>Exp Return</th>
<th>Major Up</th><th>Mildly Up</th><th>Mildly Down</th><th>Major Down</th>
<th>1D</th><th>5D</th><th>20D</th><th>RSI(14)</th>
</tr>
{rows}
</table>
<p class="disclaimer">Signals are next-day directional probability forecasts, not buy/sell recommendations. High-conviction flags are based on decision-tree leaves that exceeded {MIN_LEAF_ACCURACY:.0%} validation accuracy and are verified on an untouched test set from the last 24 months of data.</p>
</div>
</body>
</html>"""


def generate() -> dict[str, Any]:
    """Run the full pipeline and return the output summary."""
    print("Loading ETF universe...")
    universe = load_universe()
    non_lev = universe[universe["non_leveraged"]].copy()
    print(f"Non-leveraged universe: {len(non_lev)} / {len(universe)} ETFs")

    etf_tickers = non_lev["symbol"].tolist()
    names = dict(zip(non_lev["symbol"], non_lev["name"]))

    print("Downloading prices...")
    etf_data = fetch_prices(etf_tickers, period="2y")
    panel = build_panel(etf_data, etf_tickers, names)

    print("Engineering features...")
    panel = engineer_features(panel)
    panel = add_macro_features(panel)
    panel = add_cross_sectional_features(panel)

    # Require at least 66 rows of history for an ETF to be processed.
    panel = panel.groupby("Ticker").filter(lambda g: len(g) >= 66)

    panel = panel.sort_values(["Ticker", "date"])
    last_date_per_ticker = panel.groupby("Ticker")["date"].transform("max")
    live_df = panel[panel["date"] == last_date_per_ticker].copy()
    hist_df = panel[panel["date"] < last_date_per_ticker].copy()

    if hist_df.empty or live_df.empty:
        raise RuntimeError("Insufficient data for training or live prediction.")

    q25, q50, q75 = bucket_boundaries(hist_df["next_day_return_pct"])
    hist_df["bucket"] = bucket_returns(hist_df["next_day_return_pct"], q25, q50, q75)
    live_df["bucket"] = bucket_returns(live_df["next_day_return_pct"], q25, q50, q75)
    live_df["bucket"] = live_df["bucket"].fillna(-1).astype(int)

    print(f"Historical rows: {len(hist_df)}, live rows: {len(live_df)}")
    print("Training and calibrating high-conviction leaves...")
    model, leaf_map, summary, records = train_and_calibrate(hist_df, live_df)

    summary.update({
        "universe_total": int(len(universe)),
        "non_leveraged_count": int(len(non_lev)),
        "processed_count": len(records),
        "top_10": [r["symbol"] for r in records[:10]],
        "bottom_10": [r["symbol"] for r in records[-10:]],
        "bucket_quantiles": {"q25": round(q25, 6), "q50": round(q50, 6), "q75": round(q75, 6)},
        "signals": records,
    })

    json_path = OUTPUT_DIR / "latest_signals.json"
    html_path = OUTPUT_DIR / "latest_signals.html"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(html_path, "w") as f:
        f.write(build_html(records, summary))

    print(f"Signals written to {json_path} and {html_path}")
    print(f"High-conviction backtest accuracy: {summary['backtest_high_conviction_accuracy']:.2%} ({summary['backtest_high_conviction_signals']} signals)")
    print("Top 5:", summary["top_10"][:5])
    print("Bottom 5:", summary["bottom_10"][:5])

    return summary


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
