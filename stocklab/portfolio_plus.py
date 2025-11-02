import pandas as pd, numpy as np

def inverse_vol_weights(vol_row, picks):
    v = vol_row.reindex(picks.index).replace(0, np.nan)
    w = 1.0/v
    w = w.where(picks, 0.0)
    if w.sum()>0:
        w = w/w.sum()
    return w.fillna(0.0)

def sector_neutral_adjust(weights, sector_map, max_dev=0.1):
    if weights.empty:
        return weights
    sec = pd.Series(sector_map)
    W = weights.copy()
    for d in W.index:
        w = W.loc[d].copy()
        ss = w.groupby(sec.reindex(w.index)).sum().dropna()
        tgt = pd.Series(1.0/len(ss), index=ss.index)
        diff = ss - tgt
        if (diff.abs().sum()>0):
            for k in ss.index:
                names = [t for t in w.index if sec.get(t)==k]
                if names:
                    adj = -diff[k]/len(names)
                    w.loc[names] = (w.loc[names] + adj).clip(lower=0)
            if w.sum()>0:
                w = w/w.sum()
        W.loc[d] = w
    return W

def vol_target(rets, weights, target_ann_vol=0.15, span=20):
    pr = (weights.shift().fillna(0.0)*rets).sum(axis=1)
    cur_vol = pr.ewm(span=span, adjust=False).std() * (252**0.5)
    scale = target_ann_vol / cur_vol.replace(0, np.nan)
    scale = scale.clip(upper=2.0).fillna(1.0)
    W = weights.mul(scale, axis=0)
    W = W.div(W.sum(axis=1), axis=0).fillna(0.0)
    return W
