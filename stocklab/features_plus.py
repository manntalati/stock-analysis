import pandas as pd
from .indicators import returns, rolling_vol, atr, ma, slope_log_price_df, momentum_k
from .indicators_plus import rsi, macd, bollinger, pct_to_52w_high, overnight_return, ewma_vol

def _flatten(df, name):
    df = df.copy()
    df.columns = [f"{name}::{c}" for c in df.columns]
    return df

def build_features_plus(ohlc):
    px = ohlc['Close']
    hi = ohlc['High']
    lo = ohlc['Low']
    rets = returns(px)
    vol63 = rolling_vol(rets, 63)
    ewv = ewma_vol(rets, 63)
    atr14 = atr(hi, lo, ohlc['Close'], 14)
    ma50 = ma(px, 50)
    ma200 = ma(px, 200)
    slope = slope_log_price_df(px, 63)
    mom12 = momentum_k(px, 252)
    mom6 = momentum_k(px, 126)
    mom3 = momentum_k(px, 63)
    rsi14 = rsi(px, 14)
    mline, msig, mhist = macd(px, 12, 26, 9)
    bb_ma, bb_u, bb_l, bb_pctb = bollinger(px, 20, 2)
    pct52h = pct_to_52w_high(px)
    ovr = overnight_return(ohlc)
    feats = pd.concat([
        _flatten(vol63, "vol63"),
        _flatten(ewv, "ewmvol63"),
        _flatten(atr14, "atr14"),
        _flatten(slope, "slope63"),
        _flatten(mom12, "mom252"),
        _flatten(mom6, "mom126"),
        _flatten(mom3, "mom63"),
        _flatten(px.div(ma200)-1, "pctma200"),
        _flatten(ma50.div(ma200)-1, "ma50_200"),
        _flatten(rsi14, "rsi14"),
        _flatten(mline, "macd"),
        _flatten(msig, "macdsig"),
        _flatten(mhist, "macdhist"),
        _flatten(bb_pctb, "bb_pctb"),
        _flatten(pct52h, "pct_52w_high"),
        _flatten(ovr, "overnight"),
    ], axis=1).dropna()
    return feats
