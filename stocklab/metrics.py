import pandas as pd
import numpy as np

def equity_curve(rets):
    return (1+rets).cumprod()

def drawdown(curve):
    return curve/curve.cummax()-1

def perf_summary(rets):
    curve = equity_curve(rets)
    dd = drawdown(curve)
    cagr = curve.iloc[-1]**(252/len(curve))-1 if len(curve)>0 else 0.0
    vol = rets.std()*np.sqrt(252)
    downside = rets[rets<0].std()*np.sqrt(252)
    sharpe = cagr/vol if vol>0 else 0.0
    sortino = cagr/downside if downside>0 else 0.0
    mdd = dd.min() if len(dd)>0 else 0.0
    hit = (rets>0).mean()
    return {"CAGR":cagr,"Vol":vol,"Sharpe":sharpe,"Sortino":sortino,"MaxDD":mdd,"HitRate":hit}
