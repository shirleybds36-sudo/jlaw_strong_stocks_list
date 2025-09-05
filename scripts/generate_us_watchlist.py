# scripts/generate_us_watchlist.py
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime
from pathlib import Path

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

HEADERS = {
    # Pretend to be a real browser. Wikipedia blocks default Python UA.
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch_html(url: str, retries: int = 3, backoff: float = 3.0) -> str:
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            else:
                print(f"[warn] {url} -> HTTP {r.status_code}, retrying...")
        except Exception as e:
            last_exc = e
            print(f"[warn] {url} fetch error: {e}, retrying...")
        time.sleep(backoff * (i + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url}")

def fetch_constituents_from_web() -> list[str]:
    # S&P 500
    sp_html = fetch_html(SP500_URL)
    sp_tables = pd.read_html(sp_html, match="Symbol", flavor="lxml")
    sp = sp_tables[0]
    sp_tickers = (
        sp["Symbol"]
        .astype(str)
        .str.replace(".", "-", regex=False)   # BRK.B -> BRK-B (Yahoo style)
        .str.strip()
        .tolist()
    )

    # NASDAQ-100 (table layout can vary, so search for ticker-like column)
    ndx_html = fetch_html(NDX100_URL)
    ndx_tables = pd.read_html(ndx_html, flavor="lxml")
    ndx = None
    for tbl in ndx_tables:
        cols = [str(c).lower() for c in tbl.columns]
        if any("ticker" in c or "symbol" in c for c in cols):
            ndx = tbl
            break
    if ndx is None:
        ndx = ndx_tables[0]
    # find first column that looks like tickers
    col = None
    for c in ndx.columns:
        if any(k in str(c).lower() for k in ["ticker", "symbol"]):
            col = c
            break
    if col is None:
        col = ndx.columns[0]
    ndx_tickers = ndx[col].astype(str).str.strip().tolist()

    tickers = sorted(list({*sp_tickers, *ndx_tickers}))
    tickers = [t for t in tickers if t and t.upper() != "N/A"]
    return tickers

def load_fallback_csvs() -> list[str]:
    """
    Optional: if web fetch fails, try local CSV fallbacks:
      - data/sp500_fallback.csv with a 'Symbol' column
      - data/nasdaq100_fallback.csv with a 'Ticker' or 'Symbol' column
    If these files don't exist, return a minimal core list so the pipeline still runs.
    """
    base = Path("data")
    tickers = set()

    sp_csv = base / "sp500_fallback.csv"
    if sp_csv.exists():
        df = pd.read_csv(sp_csv)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        ts = df[col].astype(str).str.replace(".", "-", regex=False).str.strip()
        tickers.update(ts.tolist())

    ndx_csv = base / "nasdaq100_fallback.csv"
    if ndx_csv.exists():
        df = pd.read_csv(ndx_csv)
        # accept Ticker or Symbol
        col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else df.columns[0])
        ts = df[col].astype(str).str.strip()
        tickers.update(ts.tolist())

    # If nothing available, provide a minimal robust core so the job keeps running.
    if not tickers:
        tickers = {
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","NFLX","AVGO","COST",
            "JPM","LLY","UNH","V","MA","HD","PEP","KO","XOM","CVX"
        }
    return sorted(tickers)

def fetch_constituents() -> list[str]:
    try:
        return fetch_constituents_from_web()
    except Exception as e:
        print(f"[warn] fetch_constituents_from_web failed: {e}")
        print("[info] Falling back to local CSVs or minimal core list.")
        return load_fallback_csvs()

# --- rest of your script stays the same ---

def pct_change(s: pd.Series, n: int) -> pd.Series:
    return s / s.shift(n) - 1.0

def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "watchlist.txt"

    universe = fetch_constituents()
    # 加上基准方便对比
    all_tickers = sorted(list({*universe, "SPY"}))

    # 下载
    df = yf.download(all_tickers, period="420d", interval="1d", auto_adjust=False, progress=False, group_by="ticker")
    if not isinstance(df.columns, pd.MultiIndex):
        df = pd.concat({all_tickers[0]: df}, axis=1)

    # 参数
    SMA_SHORT, SMA_LONG = 50, 200
    H12M, H6M = 252, 126
    DOLLAR_VOL_MIN = 50_000_000  # 5000 万美元
    MAX_GAP_52W = 0.25           # 距 52周高 ≤ 25%
    RS_PCT_MIN = 70.0            # 相对强度百分位 ≥ 70

    rows = []
    for t in universe:
        if t not in df.columns.get_level_values(0):
            continue
        sub = df[t].dropna()
        if sub.empty or len(sub) < SMA_LONG + 5:
            continue

        close = sub["Adj Close"]
        sma50 = close.rolling(SMA_SHORT).mean()
        sma200 = close.rolling(SMA_LONG).mean()

        # 趋势条件
        last_close = float(close.iloc[-1])
        trend_ok = (last_close > float(sma50.iloc[-1])) and (float(sma50.iloc[-1]) >= float(sma200.iloc[-1]))

        # 52周高距离
        high52 = close.rolling(H12M, min_periods=50).max()
        gap_52w = (float(high52.iloc[-1]) - last_close) / float(high52.iloc[-1]) if float(high52.iloc[-1]) > 0 else np.nan
        high_ok = (gap_52w <= MAX_GAP_52W) if pd.notna(gap_52w) else False

        # 50日美元成交额
        vol50 = sub["Volume"].rolling(50).mean().iloc[-1]
        dollar_vol50 = float(vol50) * last_close if pd.notna(vol50) else 0.0
        liquidity_ok = dollar_vol50 >= DOLLAR_VOL_MIN

        rows.append({
            "ticker": t,
            "last_close": last_close,
            "trend_ok": trend_ok,
            "high_ok": high_ok,
            "dollar_vol50": dollar_vol50,
        })

    base = pd.DataFrame(rows)
    # 相对强度（6个月 vs SPY）
    spy = df["SPY"]["Adj Close"].dropna()
    if not base.empty and not spy.empty:
        # 统一索引
        for t in base["ticker"]:
            sub = df[t]["Adj Close"].dropna()
            idx = spy.index.intersection(sub.index)
            if len(idx) < H6M + 5:
                base.loc[base["ticker"] == t, "rs6"] = np.nan
                continue
            r6 = sub.loc[idx] / sub.loc[idx].shift(H6M) - 1.0
            br6 = spy.loc[idx] / spy.loc[idx].shift(H6M) - 1.0
            rel = (1 + r6.iloc[-1]) / (1 + br6.iloc[-1]) - 1.0
            base.loc[base["ticker"] == t, "rs6"] = float(rel)
        base["rs6_pctile"] = base["rs6"].rank(pct=True) * 100.0

    # 组合过滤
    filt = (
        base["trend_ok"].fillna(False) &
        base["high_ok"].fillna(False) &
        (base["dollar_vol50"] >= DOLLAR_VOL_MIN) &
        (base["rs6_pctile"] >= RS_PCT_MIN)
    )
    watch = base.loc[filt].sort_values(["rs6_pctile", "dollar_vol50"], ascending=[False, False])

    # 控制规模（可选）：最多取 200 只
    watch = watch.head(200)

    # 输出 watchlist
    with open(out_path, "w") as f:
        f.write("\n".join(watch["ticker"].tolist()))

    print(f"Initial US watchlist generated: {out_path} ({len(watch)} tickers)")

if __name__ == "__main__":
    main()
