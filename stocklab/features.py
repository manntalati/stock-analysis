import pandas as pd
from .indicators import returns, rolling_vol, atr, ma, slope_log_price_df, momentum_k

def _flatten(df, name):
    df = df.copy()
    df.columns = [f"{name}::{c}" for c in df.columns]
    return df

def build_features(ohlc):
    px = ohlc['Close']
    hi = ohlc['High']
    lo = ohlc['Low']
    rets = returns(px)
    vol63 = rolling_vol(rets, 63)
    atr14 = atr(hi, lo, ohlc['Close'], 14)
    ma50 = ma(px, 50)
    ma200 = ma(px, 200)
    slope = slope_log_price_df(px, 63)
    mom12 = momentum_k(px, 252)
    mom6 = momentum_k(px, 126)
    mom3 = momentum_k(px, 63)
    pct_above_200 = px.div(ma200) - 1
    rel_ma = ma50.div(ma200) - 1
    feats = pd.concat([
        _flatten(vol63, "vol63"),
        _flatten(atr14, "atr14"),
        _flatten(slope, "slope63"),
        _flatten(mom12, "mom252"),
        _flatten(mom6, "mom126"),
        _flatten(mom3, "mom63"),
        _flatten(pct_above_200, "pctma200"),
        _flatten(rel_ma, "ma50_200"),
    ], axis=1)
    feats = feats.dropna()
    return feats
