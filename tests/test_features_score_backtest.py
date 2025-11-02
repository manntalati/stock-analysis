import numpy as np, pandas as pd
from stocklab.features import build_features
from stocklab.score import composite
from stocklab.backtest import run
from stocklab.portfolio import topn_weights

def synth_ohlc(tickers=("AAA","BBB","CCC"), n=400):
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    close = pd.DataFrame({t:100*np.cumprod(1+np.random.normal(0,0.01,n)) for t in tickers}, index=idx)
    high = close*1.01
    low = close*0.99
    openp = close.shift().fillna(close)
    vol = pd.DataFrame({t:(1e6+np.random.randint(0,1e6,n)) for t in tickers}, index=idx)
    ohlc = pd.concat({"Open":openp, "High":high, "Low":low, "Close":close, "Adj Close":close, "Volume":vol}, axis=1)
    return ohlc

def test_pipeline_simple():
    ohlc = synth_ohlc()
    feats = build_features(ohlc)
    assert feats.notna().all().all()
    scores, _ = composite(feats, None)
    adv = (ohlc["Close"]*ohlc["Volume"]).rolling(20).mean()
    d = scores.index[-1]
    w = topn_weights(scores.loc[d].to_frame().T, adv.loc[d], top_n=2)
    px = ohlc["Close"]
    W = pd.DataFrame([w.iloc[0]]*50, index=scores.index[-50:], columns=scores.columns).fillna(0.0)
    res = run(px.reindex(columns=W.columns).loc[W.index], W)
    assert isinstance(res["cagr"], float)
