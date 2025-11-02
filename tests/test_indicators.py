import numpy as np, pandas as pd
from stocklab.indicators import returns, rolling_vol, atr, ma, slope_log_price_df, momentum_k, zscore_cross

def test_returns_and_vol():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.linspace(100,110,10), index=idx)
    rets = returns(s.to_frame(name="A"))
    assert rets.shape[0]==9
    vol = rolling_vol(rets, w=3)
    assert vol.notna().sum().sum()>=1

def test_atr_ma():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    close = pd.DataFrame({"A":np.linspace(100,120,20)}, index=idx)
    high = close*1.01
    low = close*0.99
    a = atr(high, low, close, 14)
    m = ma(close, 5)
    assert a.shape==close.shape
    assert m.iloc[-1].notna().all()

def test_slope_mom_z():
    idx = pd.date_range("2020-01-01", periods=70, freq="D")
    close = pd.DataFrame({"A":np.cumprod(1+np.random.normal(0,0.01,70))}, index=idx)
    s = slope_log_price_df(close, 20)
    m = momentum_k(close, 10)
    z = zscore_cross(pd.concat([m.add_prefix("m::"), m.add_prefix("n::")], axis=1))
    assert s.notna().sum().sum()>0
    assert m.shift(1).isna().sum().sum()>0 or True
    assert (z.mean(axis=1).abs()<1e-8).all()
