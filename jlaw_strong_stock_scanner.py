#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JLaw-Style Strong Stock Scanner (Free data, Yahoo Finance via yfinance)

What it does, in one pass:
1) Pulls daily OHLCV for your tickers + benchmark (default: SPY)
2) Computes multi-horizon relative strength vs benchmark (3M/6M/12M) and a composite RS percentile
3) Checks "near 52-week high", "tight pivot", "volume dry-up", and a "volatility contraction score"
4) Emits a CSV of candidates (ranked by RS), plus a full metrics CSV for all tickers
"""
import argparse
import math
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# yfinance is required. Install with: pip install yfinance
import yfinance as yf


def read_watchlist(path: str) -> List[str]:
    with open(path, 'r') as f:
        tickers = [ln.strip().upper() for ln in f if ln.strip() and not ln.strip().startswith('#')]
    # de-dup and drop benchmark if present (it will be added separately)
    tickers = sorted(list({t for t in tickers}))
    return tickers


def fetch_prices(tickers: List[str], period: str = "420d", interval: str = "1d") -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ticker-level OHLCV in a column MultiIndex
    Index is DatetimeIndex. Adjusted Close is used for returns computations.
    """
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False, group_by='ticker')
    # Normalize to MultiIndex [Date, Ticker, Field]
    if isinstance(data.columns, pd.MultiIndex):
        # already grouped
        pass
    else:
        # single ticker case -> add a level
        data = pd.concat({tickers[0]: data}, axis=1)
    return data


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def compute_metrics(data: pd.DataFrame, tickers: List[str], benchmark: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-ticker metrics:
    - Returns over 3M(63d), 6M(126d), 12M(252d)
    - RS vs benchmark on those horizons (+ composite weighted score)
    - 52w high proximity
    - Tight pivot (last N days range <= pct and within 15% of 52w high)
    - Volatility contraction score (slope of 20d ATR over last 60 trading days)
    - Volume dry-up: 10d avg vol < 50d avg vol * thresh
    - Breakout trigger: recent 20d high
    """
    # Ensure we have the columns we need
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [f for f in fields if f not in set(data.columns.get_level_values(1))]
    if missing:
        raise ValueError(f"Missing fields in downloaded data: {missing}")

    # Horizons in trading days
    H3M, H6M, H12M = 63, 126, 252

    # Containers
    rows = []

    for t in tickers:
        df = data[t].dropna().copy()
        if df.empty or len(df) < 260:
            # need ~1 year of data for robust stats
            continue

        # Returns
        r3 = _pct_change(df["Adj Close"], H3M)
        r6 = _pct_change(df["Adj Close"], H6M)
        r12 = _pct_change(df["Adj Close"], H12M)

        # Benchmark returns over same windows
        bdf = data[benchmark].reindex(df.index).dropna()
        if bdf.empty:
            continue
        br3 = _pct_change(bdf["Adj Close"], H3M)
        br6 = _pct_change(bdf["Adj Close"], H6M)
        br12 = _pct_change(bdf["Adj Close"], H12M)

        # Latest values
        if any(x.index[-1] != r3.index[-1] for x in [r3, r6, r12, br3, br6, br12]):
            # align to last index
            last_idx = min(s.dropna().index[-1] for s in [r3, r6, r12, br3, br6, br12])
            r3 = r3.reindex_like(bdf["Adj Close"]).loc[:last_idx]
            r6 = r6.reindex_like(bdf["Adj Close"]).loc[:last_idx]
            r12 = r12.reindex_like(bdf["Adj Close"]).loc[:last_idx]
            br3 = br3.loc[r3.index]
            br6 = br6.loc[r6.index]
            br12 = br12.loc[r12.index]

        r3_latest = float(r3.iloc[-1])
        r6_latest = float(r6.iloc[-1]) if not np.isnan(r6.iloc[-1]) else np.nan
        r12_latest = float(r12.iloc[-1]) if not np.isnan(r12.iloc[-1]) else np.nan

        br3_latest = float(br3.iloc[-1])
        br6_latest = float(br6.iloc[-1]) if not np.isnan(br6.iloc[-1]) else np.nan
        br12_latest = float(br12.iloc[-1]) if not np.isnan(br12.iloc[-1]) else np.nan

        # Relative performance vs benchmark
        def rel(a, b):
            if np.isnan(a) or np.isnan(b):
                return np.nan
            return (1 + a) / (1 + b) - 1.0

        rs3 = rel(r3_latest, br3_latest)
        rs6 = rel(r6_latest, br6_latest)
        rs12 = rel(r12_latest, br12_latest)

        # 52w high metrics
        close = df["Adj Close"]
        high52 = close.rolling(H12M, min_periods=50).max()
        high52_latest = float(high52.iloc[-1])
        last_close = float(close.iloc[-1])
        near_52w = (high52_latest - last_close) / high52_latest if high52_latest > 0 else np.nan

        # Tight pivot: last N days range divided by last_close <= threshold
        N_TIGHT = 7
        pivot_range = (df["High"].tail(N_TIGHT).max() - df["Low"].tail(N_TIGHT).min())
        pivot_tight = (pivot_range / last_close) if last_close > 0 else np.nan

        # ATR-based volatility contraction
        # ATR(20) over last 90d, fit slope; negative slope means contracting
        TR = pd.concat([
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs()
        ], axis=1).max(axis=1)

        atr20 = TR.rolling(20, min_periods=10).mean()
        atr_window = atr20.tail(90).dropna()
        if len(atr_window) >= 20:
            # normalize x to [0,1] for scale-invariant slope
            x = np.linspace(0.0, 1.0, len(atr_window))
            y = atr_window.values
            # linear regression via polyfit
            slope, intercept = np.polyfit(x, y, 1)
            # normalize slope by current price to make it comparable across tickers
            vol_contraction_score = slope / max(last_close, 1e-6)
        else:
            vol_contraction_score = np.nan

        # Volume dry-up
        vol10 = df["Volume"].rolling(10).mean().iloc[-1]
        vol50 = df["Volume"].rolling(50).mean().iloc[-1]
        vol_dryup_ratio = vol10 / vol50 if vol50 and not math.isnan(vol50) else np.nan

        # Breakout trigger: 20-day high
        breakout_trigger = float(df["High"].rolling(20).max().iloc[-1])

        rows.append({
            "ticker": t,
            "last_close": last_close,
            "r3": r3_latest, "r6": r6_latest, "r12": r12_latest,
            "rs3": rs3, "rs6": rs6, "rs12": rs12,
            "high52": high52_latest,
            "near_52w_gap_pct": near_52w,        # <= 0.15 preferred
            "pivot_tight_range_pct": pivot_tight, # <= 0.05 preferred
            "vol_contraction_score": vol_contraction_score,  # < 0 indicates contraction
            "vol10_over_vol50": vol_dryup_ratio, # < 0.7 indicates dry-up
            "breakout_trigger": breakout_trigger
        })

    full = pd.DataFrame(rows)

    # Composite RS score (weighted over horizons); handle NaNs safely
    def nz(x, d=0.0): return 0.0 if pd.isna(x) else x
    if not full.empty:
        comp = 0.50 * full["rs6"].apply(nz) + 0.30 * full["rs3"].apply(nz) + 0.20 * full["rs12"].apply(nz)
        full["rs_composite"] = comp

        # Percentile ranks (0-100) across universe
        for col in ["rs3", "rs6", "rs12", "rs_composite"]:
            full[f"{col}_pctile"] = full[col].rank(pct=True) * 100.0

    return full, data


def filter_candidates(metrics: pd.DataFrame,
                      rs_pctile_min: float = 90.0,
                      near_52w_max_gap: float = 0.15,
                      pivot_tight_max_range: float = 0.06,
                      vol_contraction_max: float = 0.0,
                      vol_dryup_max_ratio: float = 0.8) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    conds = [
        metrics["rs_composite_pctile"] >= rs_pctile_min,
        metrics["near_52w_gap_pct"] <= near_52w_max_gap,
        metrics["pivot_tight_range_pct"] <= pivot_tight_max_range,
        metrics["vol_contraction_score"] <= vol_contraction_max,
        metrics["vol10_over_vol50"] <= vol_dryup_max_ratio,
    ]
    mask = np.logical_and.reduce([c.fillna(False) for c in conds])
    out = metrics.loc[mask].copy()
    out = out.sort_values(["rs_composite_pctile", "rs6_pctile", "rs3_pctile"], ascending=False)
    return out


def run_scan(watchlist_file: str,
             out_dir: str,
             benchmark: str = "SPY",
             rs_pctile_min: float = 90.0,
             near_52w_max_gap: float = 0.15,
             pivot_tight_max_range: float = 0.06,
             vol_contraction_max: float = 0.0,
             vol_dryup_max_ratio: float = 0.8) -> Dict[str, str]:
    tickers = read_watchlist(watchlist_file)
    if benchmark.upper() not in tickers:
        tickers_all = sorted(tickers + [benchmark.upper()])
    else:
        tickers_all = tickers

    # Fetch data
    data = fetch_prices(tickers_all, period="420d", interval="1d")

    # Compute metrics (exclude benchmark from results)
    metrics, _ = compute_metrics(data, [t for t in tickers_all if t != benchmark.upper()], benchmark.upper())

    # Filter candidates
    cands = filter_candidates(metrics,
                              rs_pctile_min=rs_pctile_min,
                              near_52w_max_gap=near_52w_max_gap,
                              pivot_tight_max_range=pivot_tight_max_range,
                              vol_contraction_max=vol_contraction_max,
                              vol_dryup_max_ratio=vol_dryup_max_ratio)

    # Output
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    all_path = os.path.join(out_dir, f"metrics_all_{stamp}.csv")
    cands_path = os.path.join(out_dir, f"candidates_{stamp}.csv")

    metrics.to_csv(all_path, index=False)
    cands.to_csv(cands_path, index=False)

    return {"all": all_path, "candidates": cands_path}


def main():
    parser = argparse.ArgumentParser(description="Scan for strong stocks (free Yahoo Finance data).")
    parser.add_argument("-w", "--watchlist", type=str, required=True, help="Path to a text file with tickers (one per line).")
    parser.add_argument("-o", "--outdir", type=str, default="scan_outputs", help="Directory to write CSV outputs.")
    parser.add_argument("-b", "--benchmark", type=str, default="SPY", help="Benchmark ticker (default: SPY).")
    parser.add_argument("--rs_pctile_min", type=float, default=90.0, help="Minimum RS composite percentile (0-100).")
    parser.add_argument("--near_52w_max_gap", type=float, default=0.15, help="Max gap from 52w high (e.g., 0.15 = within 15%).")
    parser.add_argument("--pivot_tight_max_range", type=float, default=0.06, help="Max 7-day range/price (e.g., 0.06 = 6%).")
    parser.add_argument("--vol_contraction_max", type=float, default=0.0, help="Max (i.e., most positive) ATR slope/price (<=0 means contracting).")
    parser.add_argument("--vol_dryup_max_ratio", type=float, default=0.8, help="Max vol10/vol50 (<=0.8 means volume dry-up).")

    args = parser.parse_args()

    paths = run_scan(
        watchlist_file=args.watchlist,
        out_dir=args.outdir,
        benchmark=args.benchmark,
        rs_pctile_min=args.rs_pctile_min,
        near_52w_max_gap=args.near_52w_max_gap,
        pivot_tight_max_range=args.pivot_tight_max_range,
        vol_contraction_max=args.vol_contraction_max,
        vol_dryup_max_ratio=args.vol_dryup_max_ratio,
    )
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
