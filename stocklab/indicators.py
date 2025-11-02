import numpy as np
import pandas as pd

def returns(prices):
    return prices.pct_change().dropna()

def rolling_vol(rets, w=63):
    return rets.rolling(w).std()

def atr(high, low, close, w=14):
    if isinstance(high, pd.DataFrame):
        tr = pd.concat([
            (high-low).abs().stack(),
            (high-close.shift()).abs().stack(),
            (low-close.shift()).abs().stack()
        ], axis=1).max(1).unstack()
        return tr.rolling(w).mean()
    tr = pd.concat([(high-low).abs(), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(1)
    return tr.rolling(w).mean()

def ma(x, w):
    return x.rolling(w).mean()

def slope_log_price_df(prices, w=63):
    def slope_series(s):
        lp = np.log(s)
        idx = np.arange(w)
        def f(x):
            y = x.values
            X = np.c_[np.ones_like(idx), idx]
            b = np.linalg.lstsq(X, y, rcond=None)[0][1]
            return b
        return lp.rolling(w).apply(f, raw=False)
    return prices.apply(slope_series, axis=0)

def momentum_k(prices, k):
    return prices.pct_change(k).shift(1)

def zscore_cross(df):
    m = df.mean(axis=1)
    s = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(m, axis=0).div(s, axis=0)
