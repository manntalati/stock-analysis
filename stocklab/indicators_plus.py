import numpy as np
import pandas as pd

def ema(x, span):
    return x.ewm(span=span, adjust=False).mean()

def rsi(close, w=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rol_up = up.ewm(span=w, adjust=False).mean()
    rol_down = down.ewm(span=w, adjust=False).mean()
    rs = rol_up / (rol_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    sig = ema(macd_line, signal)
    hist = macd_line - sig
    return macd_line, sig, hist

def bollinger(close, w=20, k=2):
    ma = close.rolling(w).mean()
    sd = close.rolling(w).std()
    upper = ma + k*sd
    lower = ma - k*sd
    pct_b = (close - lower) / (upper - lower)
    return ma, upper, lower, pct_b

def pct_to_52w_high(close):
    roll_high = close.rolling(252).max()
    return close/roll_high - 1.0

def overnight_return(ohlc):
    return ohlc["Open"].div(ohlc["Close"].shift()).sub(1.0)

def ewma_vol(rets, span=63):
    v = rets.ewm(span=span, adjust=False).std()
    return v
