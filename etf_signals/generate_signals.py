#!/usr/bin/env python3
"""Generate daily momentum/risk-adjusted signals for the top 100 US-listed ETFs.

Outputs:
    output/latest_signals.json  -- machine-readable per-ticker signals
    output/latest_signals.html  -- human-readable HTML screener
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

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

HORizons = [1, 5, 10, 20, 60]


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


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """Download 1y of adjusted prices and volume for all tickers."""
    data = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    return data


def process_ticker(prices: pd.DataFrame) -> dict | None:
    """Return signal features for a single ticker DataFrame."""
    if prices is None or prices.empty:
        return None
    if len(prices) < 66:
        return None

    close = prices["Close"].dropna()
    volume = prices["Volume"].dropna() if "Volume" in prices.columns else pd.Series(dtype=float)
    if len(close) < 66:
        return None

    log_ret = np.log(close / close.shift(1)).dropna()
    if log_ret.std() == 0 or log_ret.std() != log_ret.std():
        return None

    ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1)
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1)
    ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1)
    ret_60d = float(close.iloc[-1] / close.iloc[-61] - 1)

    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    dist_20ma = float(close.iloc[-1] / ma20 - 1)
    dist_50ma = float(close.iloc[-1] / ma50 - 1)

    rsi14 = float(rsi(close, 14).iloc[-1])
    vol_20d = float(log_ret.tail(20).std() * np.sqrt(252))

    vol_ratio = np.nan
    if not volume.empty and len(volume) >= 60:
        vol_20 = volume.tail(20).mean()
        vol_60 = volume.tail(60).mean()
        if vol_60 > 0:
            vol_ratio = float(vol_20 / vol_60)

    return {
        "latest_close": float(close.iloc[-1]),
        "return_1d_pct": ret_1d * 100,
        "return_5d_pct": ret_5d * 100,
        "return_20d_pct": ret_20d * 100,
        "return_60d_pct": ret_60d * 100,
        "rsi_14": rsi14,
        "volatility_20d_annualized_pct": vol_20d * 100,
        "distance_20ma_pct": dist_20ma * 100,
        "distance_50ma_pct": dist_50ma * 100,
        "volume_ratio_20d_60d": vol_ratio,
        "data_days": int(len(close)),
    }


def compute_score(row: pd.Series) -> float:
    """Composite risk-adjusted momentum score.

    Higher = stronger bullish conviction. Lower (negative) = bearish.
    """
    momentum = (
        0.10 * row["return_1d_pct"]
        + 0.20 * row["return_5d_pct"]
        + 0.30 * row["return_20d_pct"]
        + 0.40 * row["return_60d_pct"]
    )
    trend = 0.5 * row["distance_20ma_pct"] + 0.5 * row["distance_50ma_pct"]
    vol = row["volatility_20d_annualized_pct"]
    vol_penalty = 0.0
    if vol > 0:
        vol_penalty = -0.5 * vol  # penalise noisy instruments
    rsi_adj = 0.0
    if not np.isnan(row["rsi_14"]):
        # mild mean-reversion penalty for very overbought, bonus for strong but not extreme
        if row["rsi_14"] > 75:
            rsi_adj = -2.0
        elif row["rsi_14"] > 55:
            rsi_adj = 1.0
        elif row["rsi_14"] < 30:
            rsi_adj = 2.0
        else:
            rsi_adj = -1.0
    return float(momentum + trend + vol_penalty + rsi_adj)


def bucket_from_score(score: float, scores: pd.Series) -> str:
    """Assign a four-bucket directional label based on percentile rank."""
    if scores.empty:
        return "Neutral"
    q25 = scores.quantile(0.25)
    q75 = scores.quantile(0.75)
    if score >= q75:
        return "Major Up"
    if score >= 0 and score < q75 and score >= q25:
        return "Mildly Up"
    if score < 0 and score >= q25:
        return "Mildly Down"
    return "Major Down"


def build_html(records: list[dict], generated_at: str) -> str:
    rows = ""
    for r in records:
        cls = "bull" if "Up" in r["signal"] else "bear" if "Down" in r["signal"] else "neutral"
        rows += (
            f"<tr class='{cls}'>"
            f"<td>{r['rank']}</td>"
            f"<td><b>{r['symbol']}</b></td>"
            f"<td>{r['name']}</td>"
            f"<td>{r['signal']}</td>"
            f"<td>{r['score']:.2f}</td>"
            f"<td>{r['return_1d_pct']:.2f}%</td>"
            f"<td>{r['return_5d_pct']:.2f}%</td>"
            f"<td>{r['return_20d_pct']:.2f}%</td>"
            f"<td>{r['return_60d_pct']:.2f}%</td>"
            f"<td>{r['rsi_14']:.1f}</td>"
            f"<td>{r['volatility_20d_annualized_pct']:.1f}%</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Top 100 ETF Portfolio Signals</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#0d1117; color:#c9d1d9; margin:20px; }}
table {{ border-collapse:collapse; width:100%; margin-top:20px; font-size:13px; }}
th {{ background:#21262d; color:#58a6ff; padding:10px; text-align:left; border-bottom:2px solid #30363d; }}
td {{ padding:8px 10px; border-bottom:1px solid #30363d; }}
tr:hover {{ background:#161b22; }}
.bull {{ color:#3fb950; }}
.bear {{ color:#f85149; }}
.neutral {{ color:#8b949e; }}
.meta {{ color:#8b949e; margin-bottom:10px; }}
</style>
</head>
<body>
<h2>Top 100 US-Listed Non-Leveraged ETF Signals</h2>
<p class="meta">Generated at {generated_at} UTC. Signals are percentile-ranked momentum/trend scores, not buy/sell recommendations.</p>
<table>
<tr>
<th>Rank</th><th>Ticker</th><th>Name</th><th>Signal</th><th>Score</th>
<th>1D</th><th>5D</th><th>20D</th><th>60D</th><th>RSI(14)</th><th>Vol(20d, ann)</th>
</tr>
{rows}
</table>
</body>
</html>"""


def main() -> None:
    print("Loading ETF universe...")
    universe = load_universe()
    non_lev = universe[universe["non_leveraged"]].copy()
    print(f"Non-leveraged universe: {len(non_lev)} / {len(universe)} ETFs")

    tickers = non_lev["symbol"].tolist()
    print("Downloading prices...")
    data = fetch_prices(tickers)

    results: list[dict] = []
    for symbol in tickers:
        try:
            if len(tickers) == 1:
                df = data
            else:
                df = data.get(symbol)
            if df is None or df.empty:
                continue
            features = process_ticker(df)
            if features is None:
                continue
            row = universe[universe["symbol"] == symbol].iloc[0]
            results.append({
                "symbol": symbol,
                "name": row["name"],
                **features,
            })
        except Exception as exc:
            print(f"{symbol}: skipped ({exc})")

    if not results:
        raise RuntimeError("No ETFs could be processed. Check data source / network.")

    df = pd.DataFrame(results)
    df["score"] = df.apply(compute_score, axis=1)
    df["signal"] = df["score"].apply(lambda s: bucket_from_score(s, df["score"]))
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    generated_at = datetime.now(timezone.utc).isoformat()

    records = df[[
        "rank", "symbol", "name", "signal", "score",
        "return_1d_pct", "return_5d_pct", "return_20d_pct", "return_60d_pct",
        "rsi_14", "volatility_20d_annualized_pct", "distance_20ma_pct",
        "distance_50ma_pct", "volume_ratio_20d_60d", "data_days",
    ]].to_dict(orient="records")

    summary = {
        "generated_at": generated_at,
        "universe_total": int(len(universe)),
        "non_leveraged_count": int(len(non_lev)),
        "processed_count": len(records),
        "top_10": [r["symbol"] for r in records[:10]],
        "bottom_10": [r["symbol"] for r in records[-10:]],
        "signals": records,
    }

    json_path = OUTPUT_DIR / "latest_signals.json"
    html_path = OUTPUT_DIR / "latest_signals.html"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(html_path, "w") as f:
        f.write(build_html(records, generated_at))

    print(f"Signals written to {json_path} and {html_path}")
    print("Top 5:", summary["top_10"][:5])
    print("Bottom 5:", summary["bottom_10"][:5])


if __name__ == "__main__":
    main()
