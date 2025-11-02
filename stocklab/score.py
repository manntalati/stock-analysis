import pandas as pd
from .indicators import zscore_cross

def _pick(features, name):
    return features.filter(regex=f"^{name}::")

def composite(features, preds=None, w_mom=0.5, w_trend=0.25, w_risk=0.15, w_pred=0.10):
    mom = zscore_cross(_pick(features, "mom252")) + 0.5*zscore_cross(_pick(features, "mom126")) + 0.5*zscore_cross(_pick(features, "mom63"))
    trend = zscore_cross(_pick(features, "pctma200")) + zscore_cross(_pick(features, "ma50_200")) + zscore_cross(_pick(features, "slope63"))
    risk = -zscore_cross(_pick(features, "vol63")) - zscore_cross(_pick(features, "atr14"))
    parts = [w_mom*mom, w_trend*trend, w_risk*risk]
    if preds is not None:
        preds_cols = [f"pred::{c}" for c in preds.columns]
        p = preds.copy()
        p.columns = preds_cols
        parts.append(w_pred*zscore_cross(p))
    comp = sum(parts)
    comp.columns = [c.split("::",1)[1] for c in comp.columns]
    s = comp.groupby(level=0, axis=1).sum()
    return s, {"mom":mom, "trend":trend, "risk":risk}
