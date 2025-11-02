import numpy as np, pandas as pd
from stocklab.features import build_features
from stocklab.predict import fit_predict, forward_return

def synth_prices(tickers=("AAA","BBB"), n=300):
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    close = pd.DataFrame({t:100*np.cumprod(1+np.random.normal(0,0.01,n)) for t in tickers}, index=idx)
    high = close*1.01
    low = close*0.99
    openp = close.shift().fillna(close)
    vol = pd.DataFrame({t:(1e6+np.random.randint(0,1e6,n)) for t in tickers}, index=idx)
    ohlc = pd.concat({"Open":openp, "High":high, "Low":low, "Close":close, "Adj Close":close, "Volume":vol}, axis=1)
    return ohlc

def test_fit_predict_shapes():
    ohlc = synth_prices()
    feats = build_features(ohlc)
    px = ohlc["Close"]
    preds = fit_predict(feats, px, h=5, n_splits=3)
    assert set(preds.columns)==set(px.columns)
    assert preds.index.min()>=feats.index.min()
