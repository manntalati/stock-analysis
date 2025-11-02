import pandas as pd
import numpy as np

def apply_tc(port_rets, w_prev, w_next, tc_bps=5):
    turn = (w_next.sub(w_prev).abs().sum(axis=1)).fillna(0.0)
    slip = turn*(tc_bps/10000.0)
    return port_rets.sub(slip, axis=0)

def run(prices, weights, tc_bps=5):
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    port_rets = (weights.shift().fillna(0.0)*rets).sum(axis=1)
    w_prev = weights.shift().fillna(0.0)
    w_next = weights.fillna(0.0)
    port_rets = apply_tc(port_rets.to_frame("r"), w_prev, w_next, tc_bps)["r"]
    curve = (1+port_rets).cumprod()
    dd = curve/curve.cummax()-1
    cagr = curve.iloc[-1]**(252/len(curve))-1 if len(curve)>0 else 0.0
    vol = port_rets.std()*np.sqrt(252)
    downside = port_rets[port_rets<0].std()*np.sqrt(252)
    sharpe = cagr/vol if vol>0 else 0.0
    sortino = cagr/downside if downside>0 else 0.0
    mdd = dd.min() if len(dd)>0 else 0.0
    trades = (w_next.sub(w_prev).abs().sum(axis=1)).sum()
    return {"curve":curve,"rets":port_rets,"sharpe":sharpe,"sortino":sortino,"cagr":cagr,"vol":vol,"maxdd":mdd,"turnover":trades}
