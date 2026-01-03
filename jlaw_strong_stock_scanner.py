#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JLaw-Style Strong Stock Scanner (Free data, Yahoo Finance via yfinance)

Outputs:
- scan_outputs/metrics_all_{TRADE_DATE}.csv
- scan_outputs/candidates_{TRADE_DATE}.csv
- scan_outputs/breakouts_{TRADE_DATE}.csv
- scan_outputs/jlaw_scanner_metrics_field_guide.csv
- data/jlaw_watchlist.txt   <-- NEW: clean candidates watchlist (one ticker per line)

Key behaviors:
- Uses benchmark last trading day as TRADE_DATE for file names.
- Robust to per-ticker download/data issues (delisted/404) without failing the whole run.
"""

import argparse
import math
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# I/O helpers
# -----------------------------
def read_watchlist(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        tickers = [ln.strip().upper() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    tickers = sorted(list({t for t in tickers if t}))
    return tickers


def fetch_prices(tickers: List[str], period: str = "420d", interval: str = "1d") -> pd.DataFrame:
    """
    Returns MultiIndex columns: (ticker, field), index: DatetimeIndex.
    Uses auto_adjust=False so we can use Adj Close for returns calculations.
    """
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if not isinstance(data.columns, pd.MultiIndex):
        # Single ticker case: wrap to MultiIndex
        data = pd.concat({tickers[0]: data}, axis=1)
    return data


def last_trade_date_from_benchmark(data: pd.DataFrame, benchmark: str) -> str:
    """
    Use benchmark's last available trading day as the stamp: YYYY-MM-DD
    """
    try:
        b = data[benchmark]["Adj Close"].dropna()
        if len(b) > 0:
            return pd.to_datetime(b.index[-1]).strftime("%Y-%m-%d")
    except Exception:
        pass
    # fallback
    return datetime.now().strftime("%Y-%m-%d")


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


# -----------------------------
# Core computations
# -----------------------------
def compute_metrics(data: pd.DataFrame, tickers: List[str], benchmark: str) -> pd.DataFrame:
    """
    Compute per-ticker metrics:
    - Returns over 3M(63d), 6M(126d), 12M(252d)
    - RS vs benchmark on those horizons (+ composite weighted score)
    - 52w high proximity
    - Tight pivot (last 7 days range/price)
    - Volatility contraction score (slope of 20d ATR over last 90 days, normalized by price)
    - Volume dry-up ratio: SMA10(vol)/SMA50(vol)
    - Breakout trigger: recent 20d highest high
    - over_trigger_high / over_trigger_close flags
    - pivot_low7 (7d low) for suggested stop
    """
    required_fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    # MultiIndex: (ticker, field)
    fields_present = set(data.columns.get_level_values(1)) if isinstance(data.columns, pd.MultiIndex) else set(data.columns)
    missing = [f for f in required_fields if f not in fields_present]
    if missing:
        raise ValueError(f"Missing fields in downloaded data: {missing}")

    H3M, H6M, H12M = 63, 126, 252
    rows = []

    # benchmark series
    try:
        bdf_all = data[benchmark].dropna().copy()
    except Exception as e:
        raise RuntimeError(f"Benchmark {benchmark} not found in downloaded data: {e}")

    for t in tickers:
        try:
            df = data[t].dropna().copy()
        except Exception:
            continue

        if df.empty or len(df) < 260:
            continue

        # Align to benchmark calendar
        bdf = bdf_all.reindex(df.index).dropna()
        if bdf.empty or len(bdf) < 260:
            continue

        # Returns
        r3 = _pct_change(df["Adj Close"], H3M)
        r6 = _pct_change(df["Adj Close"], H6M)
        r12 = _pct_change(df["Adj Close"], H12M)

        br3 = _pct_change(bdf["Adj Close"], H3M)
        br6 = _pct_change(bdf["Adj Close"], H6M)
        br12 = _pct_change(bdf["Adj Close"], H12M)

        # Ensure last index alignment
        try:
            last_idx = min(
                s.dropna().index[-1] for s in [r3, r6, r12, br3, br6, br12] if len(s.dropna()) > 0
            )
        except Exception:
            continue

        r3 = r3.loc[:last_idx]
        r6 = r6.loc[:last_idx]
        r12 = r12.loc[:last_idx]
        br3 = br3.loc[:last_idx]
        br6 = br6.loc[:last_idx]
        br12 = br12.loc[:last_idx]

        if any(len(s.dropna()) == 0 for s in [r3, br3]):
            continue

        r3_latest = float(r3.dropna().iloc[-1])
        r6_latest = float(r6.dropna().iloc[-1]) if len(r6.dropna()) else np.nan
        r12_latest = float(r12.dropna().iloc[-1]) if len(r12.dropna()) else np.nan

        br3_latest = float(br3.dropna().iloc[-1])
        br6_latest = float(br6.dropna().iloc[-1]) if len(br6.dropna()) else np.nan
        br12_latest = float(br12.dropna().iloc[-1]) if len(br12.dropna()) else np.nan

        def rel(a, b):
            if np.isnan(a) or np.isnan(b):
                return np.nan
            return (1.0 + a) / (1.0 + b) - 1.0

        rs3 = rel(r3_latest, br3_latest)
        rs6 = rel(r6_latest, br6_latest)
        rs12 = rel(r12_latest, br12_latest)

        close = df["Adj Close"].loc[:last_idx].dropna()
        if close.empty:
            continue

        last_close = float(close.iloc[-1])

        # 52w high gap
        high52 = close.rolling(H12M, min_periods=50).max()
        high52_latest = float(high52.iloc[-1]) if len(high52.dropna()) else np.nan
        near_52w_gap = (high52_latest - last_close) / high52_latest if (high52_latest and high52_latest > 0) else np.nan

        # Tight pivot: 7d range / price
        N_TIGHT = 7
        sub = df.loc[:last_idx].dropna()
        if len(sub) < N_TIGHT + 5:
            continue
        last7_high = float(sub["High"].tail(N_TIGHT).max())
        last7_low = float(sub["Low"].tail(N_TIGHT).min())
        pivot_range = last7_high - last7_low
        pivot_tight = (pivot_range / last_close) if last_close > 0 else np.nan

        # ATR contraction slope
        TR = pd.concat(
            [
                (sub["High"] - sub["Low"]).abs(),
                (sub["High"] - sub["Close"].shift(1)).abs(),
                (sub["Low"] - sub["Close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr20 = TR.rolling(20, min_periods=10).mean()
        atr_window = atr20.tail(90).dropna()
        if len(atr_window) >= 20:
            x = np.linspace(0.0, 1.0, len(atr_window))
            y = atr_window.values
            slope, _ = np.polyfit(x, y, 1)
            vol_contraction_score = slope / max(last_close, 1e-6)
        else:
            vol_contraction_score = np.nan

        # Volume dry-up ratio
        vol10 = sub["Volume"].rolling(10).mean().iloc[-1]
        vol50 = sub["Volume"].rolling(50).mean().iloc[-1]
        vol_dryup_ratio = (vol10 / vol50) if (vol50 and not math.isnan(vol50)) else np.nan

        # Breakout trigger
        breakout_trigger = float(sub["High"].rolling(20).max().iloc[-1])
        hi_today = float(sub["High"].iloc[-1])
        close_today = float(sub["Close"].iloc[-1])

        over_trigger_high = hi_today >= breakout_trigger - 1e-9
        over_trigger_close = close_today >= breakout_trigger - 1e-9

        vol_today = float(sub["Volume"].iloc[-1])
        vol50_avg = float(sub["Volume"].rolling(50).mean().iloc[-1])

        rows.append(
            {
                "ticker": t,
                "last_close": last_close,
                "r3": r3_latest,
                "r6": r6_latest,
                "r12": r12_latest,
                "rs3": rs3,
                "rs6": rs6,
                "rs12": rs12,
                "high52": high52_latest,
                "near_52w_gap_pct": near_52w_gap,
                "pivot_tight_range_pct": pivot_tight,
                "vol_contraction_score": vol_contraction_score,
                "vol10_over_vol50": vol_dryup_ratio,
                "breakout_trigger": breakout_trigger,
                "vol_today": vol_today,
                "vol50_avg": vol50_avg,
                "over_trigger_high": over_trigger_high,
                "over_trigger_close": over_trigger_close,
                "pivot_low7": last7_low,
            }
        )

    full = pd.DataFrame(rows)

    # Composite RS + percentiles
    def nz(x, d=0.0):
        return d if pd.isna(x) else x

    if not full.empty:
        full["rs_composite"] = 0.50 * full["rs6"].apply(nz) + 0.30 * full["rs3"].apply(nz) + 0.20 * full["rs12"].apply(nz)
        for col in ["rs3", "rs6", "rs12", "rs_composite"]:
            full[f"{col}_pctile"] = full[col].rank(pct=True) * 100.0

    return full


def filter_candidates(
    metrics: pd.DataFrame,
    rs_pctile_min: float = 90.0,
    near_52w_max_gap: float = 0.15,
    pivot_tight_max_range: float = 0.06,
    vol_contraction_max: float = 0.0,
    vol_dryup_max_ratio: float = 0.8,
) -> pd.DataFrame:
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


def write_field_guide_csv(out_dir: str) -> str:
    rows = [
        ("ticker","both","Stock symbol (Yahoo format).","","","Identifier used across outputs."),
        ("last_close","both","Most recent adjusted close price.","AdjClose[today]","","Reference price for risk and distance calculations."),
        ("r3","both","3-month absolute return.","AdjClose[t]/AdjClose[t-63]-1","","Momentum over ~63 trading days."),
        ("r6","both","6-month absolute return.","AdjClose[t]/AdjClose[t-126]-1","","Medium-term momentum (~126 trading days)."),
        ("r12","both","12-month absolute return.","AdjClose[t]/AdjClose[t-252]-1","","Long-term momentum (~252 trading days)."),
        ("rs3","both","3-month excess return vs benchmark.","(1+r3)/(1+bench_r3)-1","> 0 indicates outperformance","Short-term relative strength."),
        ("rs6","both","6-month excess return vs benchmark.","(1+r6)/(1+bench_r6)-1","> 0 preferred","Core weight in composite (50%)."),
        ("rs12","both","12-month excess return vs benchmark.","(1+r12)/(1+bench_r12)-1","> 0 preferred","Confirms sustained leadership."),
        ("high52","both","52-week rolling max of adj close.","max(AdjClose, window=252)","","Used to judge proximity to highs."),
        ("near_52w_gap_pct","both","Percent below 52-week high.","(high52-last_close)/high52","≤ 0.15–0.20","Focus on names near highs."),
        ("pivot_tight_range_pct","both","7-day high-low range as % of price.","(max(High,7)-min(Low,7))/last_close","≤ 0.05–0.10","VCP-style tightness; smaller = better."),
        ("vol_contraction_score","both","Slope of 20d ATR over ~90d, normalized by price.","slope(ATR20[last~90])/last_close","≤ 0 (contracting)","Negative = volatility contracting."),
        ("vol10_over_vol50","both","10-day avg volume / 50-day avg volume.","SMA10(Vol)/SMA50(Vol)","≤ 0.8–0.95","Lower means supply/dry-up."),
        ("breakout_trigger","both","20-day highest high; breakout trigger.","max(High, window=20)","","Use as over-line price."),
        ("rs_composite","both","Weighted composite of RS.","0.5*rs6 + 0.3*rs3 + 0.2*rs12","higher is stronger","Rank multiple horizons in one score."),
        ("rs_composite_pctile","both","Percentile rank of rs_composite.","rank_pct(rs_composite)","80–90+","Top tier leadership."),
        ("vol_today","both","Today’s raw share volume.","Vol[today]","","Used for surge check."),
        ("vol50_avg","both","50-day avg share volume.","SMA50(Vol)","","Baseline for volume surge."),
        ("over_trigger_high","both","Did today’s HIGH clear trigger?","High[today] ≥ breakout_trigger","True/False","Intraday over-line check."),
        ("over_trigger_close","both","Did today’s CLOSE clear trigger?","Close[today] ≥ breakout_trigger","True/False","Close confirmation (stricter)."),
        ("pivot_low7","both","7-day lowest low (pivot area low).","min(Low,7)","","Reference for suggested stop."),
        ("trigger_price","both","Suggested trigger price (copy of breakout_trigger).","breakout_trigger","","Entry only if price clears with volume."),
        ("stop_suggest","both","Suggested stop (pivot low).","pivot_low7","","Invalidation line."),
        ("risk_pct","both","Relative risk box height.","(trigger_price-stop_suggest)/trigger_price","~0.03–0.10","Position sizing uses it."),
        ("dist_to_trigger_pct","both","Distance below trigger.","(trigger_price-last_close)/trigger_price","≤ 0.02 near","How close price is to go-time."),
        ("vol_surge_mult","both","Volume surge multiple vs 50d avg.","Vol[today]/SMA50(Vol)","≥ 1.5","Breakout sponsorship."),
        ("signal","both","Action label.","Rule-based label","","BREAKOUT_CONFIRMED / OVER_TRIGGER_WEAK_VOLUME / WATCH_NEAR_TRIGGER / WATCH"),
        ("max_entry_price","breakouts","Max allowed entry above trigger.","trigger_price*(1+entry_extension_pct)","default 1.5–2%","Avoid chasing too far."),
    ]
    df = pd.DataFrame(rows, columns=[
        "metric","present_in","definition","formula_or_source","typical_thresholds","interpretation_and_usage"
    ])
    path = os.path.join(out_dir, "jlaw_scanner_metrics_field_guide.csv")
    df.to_csv(path, index=False)
    return path


def write_jlaw_watchlist(
    out_path: str,
    df: pd.DataFrame,
    trade_date: str,
    max_tickers: int,
    signals_keep: List[str],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if df is None or df.empty:
        content = [f"# jlaw_watchlist (empty) trade_date={trade_date}"]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")
        return

    d = df.copy()
    if "signal" in d.columns and signals_keep:
        d = d[d["signal"].isin(signals_keep)].copy()

    # 排序：优先 RS 最高
    sort_cols = [c for c in ["rs_composite_pctile", "rs6_pctile", "rs3_pctile"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols, ascending=False)

    tickers = d["ticker"].astype(str).str.upper().dropna().unique().tolist()
    tickers = tickers[:max_tickers]

    lines = [
        f"# jlaw_watchlist trade_date={trade_date}",
        f"# signals_keep={signals_keep}",
        f"# max_tickers={max_tickers}",
    ] + tickers

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------
# Runner
# -----------------------------
def run_scan(
    watchlist_file: str,
    out_dir: str,
    benchmark: str = "SPY",
    rs_pctile_min: float = 90.0,
    near_52w_max_gap: float = 0.15,
    pivot_tight_max_range: float = 0.06,
    vol_contraction_max: float = 0.0,
    vol_dryup_max_ratio: float = 0.8,
    vol_breakout_mult: float = 1.5,
    near_trigger_pct: float = 0.02,
    breakout_confirm_on: str = "high",
    entry_extension_pct: float = 0.02,
    # NEW watchlist output
    watchlist_out: str = "data/jlaw_watchlist.txt",
    watchlist_source: str = "candidates",  # candidates|breakouts
    watchlist_max: int = 200,
    watchlist_signals: Optional[List[str]] = None,
) -> Dict[str, str]:

    tickers = read_watchlist(watchlist_file)
    bench = benchmark.upper()
    tickers_all = sorted(set(tickers + [bench]))

    # 1) fetch
    data = fetch_prices(tickers_all, period="420d", interval="1d")
    trade_date = last_trade_date_from_benchmark(data, bench)

    # 2) metrics
    metrics = compute_metrics(data, [t for t in tickers_all if t != bench], bench)

    # universal entry/risk fields for metrics_all
    metrics = metrics.copy()
    metrics["trigger_price"] = metrics["breakout_trigger"]
    metrics["stop_suggest"] = metrics["pivot_low7"]

    tp = metrics["trigger_price"]
    st = metrics["stop_suggest"]
    lc = metrics["last_close"]
    metrics["risk_pct"] = np.where((tp > 0) & (tp > st), (tp - st) / tp, np.nan)
    metrics["max_entry_price"] = tp * (1.0 + entry_extension_pct)

    # 3) candidates filter
    cands = filter_candidates(
        metrics,
        rs_pctile_min=rs_pctile_min,
        near_52w_max_gap=near_52w_max_gap,
        pivot_tight_max_range=pivot_tight_max_range,
        vol_contraction_max=vol_contraction_max,
        vol_dryup_max_ratio=vol_dryup_max_ratio,
    )

    # 4) outputs
    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, f"metrics_all_{trade_date}.csv")
    cands_path = os.path.join(out_dir, f"candidates_{trade_date}.csv")
    brk_path = os.path.join(out_dir, f"breakouts_{trade_date}.csv")
    guide_path = os.path.join(out_dir, "jlaw_scanner_metrics_field_guide.csv")

    metrics.to_csv(all_path, index=False)

    # 5) enrich candidates + label signals
    if not cands.empty:
        c = cands.copy()
        c["trigger_price"] = c["breakout_trigger"]
        c["stop_suggest"] = c["pivot_low7"]
        c["risk_pct"] = np.where((c["trigger_price"] > 0) & (c["trigger_price"] > c["stop_suggest"]),
                                 (c["trigger_price"] - c["stop_suggest"]) / c["trigger_price"],
                                 np.nan)
        c["dist_to_trigger_pct"] = (c["trigger_price"] - c["last_close"]) / c["trigger_price"]

        c["over_trigger"] = c["over_trigger_close"] if breakout_confirm_on.lower() == "close" else c["over_trigger_high"]
        c["vol_surge_mult"] = c["vol_today"] / c["vol50_avg"]
        c["vol_surge"] = c["vol_surge_mult"] >= vol_breakout_mult

        c["max_entry_price"] = c["trigger_price"] * (1.0 + entry_extension_pct)

        def _signal(r):
            if bool(r["over_trigger"]) and bool(r["vol_surge"]):
                return "BREAKOUT_CONFIRMED"
            if bool(r["over_trigger"]) and (not bool(r["vol_surge"])):
                return "OVER_TRIGGER_WEAK_VOLUME"
            if (not bool(r["over_trigger"])) and (float(r["dist_to_trigger_pct"]) <= near_trigger_pct):
                return "WATCH_NEAR_TRIGGER"
            return "WATCH"

        c["signal"] = c.apply(_signal, axis=1)

        # write candidates
        c.to_csv(cands_path, index=False)

        # confirmed breakouts
        brk = c.loc[c["signal"] == "BREAKOUT_CONFIRMED"].copy()
        if not brk.empty:
            brk.to_csv(brk_path, index=False)
        else:
            c.head(0).to_csv(brk_path, index=False)

        # NEW: write jlaw_watchlist.txt
        if watchlist_signals is None:
            # 默认：更适合“隔天强势形态候选池”
            watchlist_signals = ["WATCH_NEAR_TRIGGER", "BREAKOUT_CONFIRMED"]

        source_df = brk if watchlist_source.lower() == "breakouts" else c
        write_jlaw_watchlist(
            out_path=watchlist_out,
            df=source_df,
            trade_date=trade_date,
            max_tickers=watchlist_max,
            signals_keep=watchlist_signals,
        )
    else:
        # empty headers
        metrics.head(0).to_csv(cands_path, index=False)
        metrics.head(0).to_csv(brk_path, index=False)
        if watchlist_signals is None:
            watchlist_signals = ["WATCH_NEAR_TRIGGER", "BREAKOUT_CONFIRMED"]
        write_jlaw_watchlist(
            out_path=watchlist_out,
            df=metrics.head(0),
            trade_date=trade_date,
            max_tickers=watchlist_max,
            signals_keep=watchlist_signals,
        )

    # field guide (always)
    try:
        guide_path = write_field_guide_csv(out_dir)
    except Exception:
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write("metric,present_in,definition,formula_or_source,typical_thresholds,interpretation_and_usage\n")

    return {
        "all": all_path,
        "candidates": cands_path,
        "breakouts": brk_path,
        "field_guide": guide_path,
        "jlaw_watchlist": watchlist_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Scan for strong stocks (free Yahoo Finance data).")
    parser.add_argument("-w", "--watchlist", type=str, required=True, help="Path to a text file with tickers (one per line).")
    parser.add_argument("-o", "--outdir", type=str, default="scan_outputs", help="Directory to write CSV outputs.")
    parser.add_argument("-b", "--benchmark", type=str, default="SPY", help="Benchmark ticker (default: SPY).")

    parser.add_argument("--rs_pctile_min", type=float, default=90.0, help="Minimum RS composite percentile (0-100).")
    parser.add_argument("--near_52w_max_gap", type=float, default=0.15, help="Max gap from 52w high (e.g., 0.15 = within 15%).")
    parser.add_argument("--pivot_tight_max_range", type=float, default=0.06, help="Max 7-day range/price (e.g., 0.06 = 6%).")
    parser.add_argument("--vol_contraction_max", type=float, default=0.0, help="Max ATR slope/price (<=0 means contracting).")
    parser.add_argument("--vol_dryup_max_ratio", type=float, default=0.8, help="Max vol10/vol50 (<=0.8 means dry-up).")

    parser.add_argument("--vol_breakout_mult", type=float, default=1.5, help="Volume surge threshold vs 50d avg volume (default 1.5x).")
    parser.add_argument("--near_trigger_pct", type=float, default=0.02, help="Distance to trigger considered 'near' (default 2%).")
    parser.add_argument("--breakout_confirm_on", choices=["high", "close"], default="high", help="Define over-trigger by HIGH or CLOSE.")
    parser.add_argument("--entry_extension_pct", type=float, default=0.015, help="Max percent above trigger allowed entry (default 1.5%).")

    # NEW watchlist output args
    parser.add_argument("--watchlist_out", type=str, default="data/jlaw_watchlist.txt", help="Path to write JLaw watchlist txt.")
    parser.add_argument("--watchlist_source", choices=["candidates", "breakouts"], default="candidates",
                        help="Use candidates or confirmed breakouts to build watchlist.")
    parser.add_argument("--watchlist_max", type=int, default=200, help="Max tickers in jlaw_watchlist.")
    parser.add_argument("--watchlist_signals", type=str, default="WATCH_NEAR_TRIGGER,BREAKOUT_CONFIRMED",
                        help="Comma-separated signals to keep for watchlist (default: WATCH_NEAR_TRIGGER,BREAKOUT_CONFIRMED).")

    args = parser.parse_args()
    signals_keep = [s.strip() for s in args.watchlist_signals.split(",") if s.strip()]

    paths = run_scan(
        watchlist_file=args.watchlist,
        out_dir=args.outdir,
        benchmark=args.benchmark,
        rs_pctile_min=args.rs_pctile_min,
        near_52w_max_gap=args.near_52w_max_gap,
        pivot_tight_max_range=args.pivot_tight_max_range,
        vol_contraction_max=args.vol_contraction_max,
        vol_dryup_max_ratio=args.vol_dryup_max_ratio,
        vol_breakout_mult=args.vol_breakout_mult,
        near_trigger_pct=args.near_trigger_pct,
        breakout_confirm_on=args.breakout_confirm_on,
        entry_extension_pct=args.entry_extension_pct,
        watchlist_out=args.watchlist_out,
        watchlist_source=args.watchlist_source,
        watchlist_max=args.watchlist_max,
        watchlist_signals=signals_keep,
    )

    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

