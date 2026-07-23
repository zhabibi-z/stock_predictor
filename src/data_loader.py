"""Stage 1 — Data Ingestion.

Downloads split/dividend-adjusted OHLCV history from Yahoo Finance.
A local CSV cache is written on first download so subsequent runs are instant.
"""

import os

import pandas as pd
import yfinance as yf


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df: pd.DataFrame = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError(f"yfinance returned no data for '{ticker}' ({start} → {end}).")
    df.dropna(inplace=True)
    print(f"      Ticker : {ticker}")
    print(f"      Rows   : {len(df):,}")
    print(f"      Range  : {df.index[0].date()}  →  {df.index[-1].date()}")
    return df


def load_or_download(ticker: str, start: str, end: str, data_dir: str = "data") -> pd.DataFrame:
    """Return a locally cached CSV when available; download and cache otherwise.

    The cache key includes the ticker AND the requested date range, so changing any of
    ticker/start/end in config.yaml uses a distinct cache file rather than silently reusing
    stale data downloaded for a different window.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{ticker}_{start}_{end}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"      Loaded from cache : {csv_path}  ({len(df):,} rows)")
        return df
    df = download_stock_data(ticker, start, end)
    df.to_csv(csv_path)
    print(f"      Cached to         : {csv_path}")
    return df
