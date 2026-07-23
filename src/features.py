"""Stage 2 — Feature Engineering.

Leakage-prevention contract
────────────────────────────
Every feature on row T is derived ONLY from OHLCV data available through
the close of day T (strictly historical, no peeking forward).

The label at row T encodes whether Close[T+1] > Close[T] — the next-day
direction we are predicting.  It is stored as "Target" and is NEVER fed
as a model input.

"Next_Close" stores Close[T+1] and serves only as the regression target
for the Ridge model.  "Fwd_Return" is the realized forward return used
exclusively by the backtesting engine.
"""

import pandas as pd


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Wilder's RSI via exponential moving average smoothing.
    com = window − 1  ↔  α = 1/window (Wilder's convention).
    min_periods=window ensures the first `window` rows are NaN so no
    partially-warmed values leak into the model.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_bollinger(
    series: pd.Series,
    window: int    = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Standard Bollinger Bands.

    %B = (Close − Lower) / (Upper − Lower) ∈ [0, 1] within the bands;
    outside-band readings carry signal about breakouts.
    """
    mean  = series.rolling(window, min_periods=window).mean()
    std   = series.rolling(window, min_periods=window).std()
    upper = mean + num_std * std
    lower = mean - num_std * std
    pct_b = (series - lower) / (upper - lower + 1e-10)
    return pd.DataFrame({"BB_Upper": upper, "BB_Lower": lower, "BB_Pct": pct_b})


def _compute_macd(
    series: pd.Series,
    fast:   int = 12,
    slow:   int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.

    min_periods=slow on both EMAs guarantees the first `slow` rows are NaN,
    preventing partially-warmed short-EMA values from corrupting MACD_Line.
    """
    ema_fast    = series.ewm(span=fast,   adjust=False, min_periods=slow).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False, min_periods=slow).mean()
    macd_line   = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist   = macd_line - macd_signal
    return pd.DataFrame({
        "MACD_Line":   macd_line,
        "MACD_Signal": macd_signal,
        "MACD_Hist":   macd_hist,
    })


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range via Wilder EMA smoothing.

    True Range = max(High−Low, |High−PrevClose|, |Low−PrevClose|).
    Normalise by Close in the caller to get a dimensionless volatility measure.
    """
    high       = df["High"]
    low        = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=window - 1, min_periods=window).mean()


def engineer_features(
    df:          pd.DataFrame,
    sma_short:   int   = 5,
    sma_long:    int   = 20,
    rsi_window:  int   = 14,
    bb_window:   int   = 20,
    bb_std:      float = 2.0,
    macd_fast:   int   = 12,
    macd_slow:   int   = 26,
    macd_sig:    int   = 9,
    atr_window:  int   = 14,
    vol_window:  int   = 20,
    lag_periods: list  = None,
) -> pd.DataFrame:
    """
    Construct the full feature matrix and labels from raw OHLCV data.

    Three columns are produced but excluded from FEATURE_COLS:
      Next_Close  — regression target for Ridge
      Target      — binary classification label (never a model input)
      Fwd_Return  — realized forward return for the backtester

    dropna() removes the indicator warm-up period and the final row
    where Next_Close / Fwd_Return are undefined.
    """
    if lag_periods is None:
        lag_periods = [1, 2, 5]

    data  = df.copy()
    close = data["Close"]

    # shift(-1) is forward-looking — label only, never a model input
    data["Next_Close"] = close.shift(-1)
    data["Target"]     = (data["Next_Close"] > close).astype(int)
    data["Fwd_Return"] = (data["Next_Close"] - close) / close

    data["Daily_Return"] = close.pct_change()
    data["SMA_5"]        = close.rolling(window=sma_short, min_periods=sma_short).mean()
    data["SMA_20"]       = close.rolling(window=sma_long,  min_periods=sma_long).mean()
    data["RSI"]          = _compute_rsi(close, window=rsi_window)

    bb = _compute_bollinger(close, window=bb_window, num_std=bb_std)
    data["BB_Upper"] = bb["BB_Upper"]
    data["BB_Lower"] = bb["BB_Lower"]
    data["BB_Pct"]   = bb["BB_Pct"]

    macd = _compute_macd(close, fast=macd_fast, slow=macd_slow, signal=macd_sig)
    data["MACD_Line"]   = macd["MACD_Line"]
    data["MACD_Signal"] = macd["MACD_Signal"]
    data["MACD_Hist"]   = macd["MACD_Hist"]

    # ATR normalised by close to make it dimensionless across all price levels
    atr = _compute_atr(df, window=atr_window)
    data["ATR"]      = atr
    data["ATR_Norm"] = atr / (close + 1e-10)

    # Volume relative to its rolling mean — > 1 signals above-average participation
    vol_ma = data["Volume"].rolling(window=vol_window, min_periods=vol_window).mean()
    data["Volume_Ratio"] = data["Volume"] / (vol_ma + 1e-10)

    for lag in lag_periods:
        data[f"Return_t{lag}"] = data["Daily_Return"].shift(lag)

    data.dropna(inplace=True)
    return data
