import pandas as pd
import numpy as np

def inverse_vol_weights(vol_row, picks):
    v = vol_row.reindex(picks.index).replace(0, np.nan)
    w = 1.0/v
    w = w.where(picks, 0.0)
    if w.sum()>0:
        w = w/w.sum()
    return w.fillna(0.0)

def topn_weights(scores, adv, top_n=20, max_w=0.08, min_adv_usd=1_000_000, method="equal", vol_ref=None):
    if isinstance(adv, pd.Series):
        adv = adv.reindex(scores.columns).fillna(0.0)
        liq = adv.gt(min_adv_usd)
        s = scores.copy()
        s = s.loc[:, liq]
        r = s.rank(axis=1, ascending=False, method="first")
        picks = r.le(top_n)
        if method=="equal":
            w = picks.div(picks.sum(axis=1), axis=0).fillna(0.0)
        else:
            W = []
            for d in picks.index:
                row = picks.loc[d]
                if vol_ref is None:
                    W.append((row.astype(float)/row.sum()) if row.sum()>0 else row.astype(float))
                else:
                    w = inverse_vol_weights(vol_ref.loc[d], row)
                    W.append(w)
            w = pd.DataFrame(W, index=picks.index, columns=picks.columns).fillna(0.0)
        w = w.clip(upper=max_w)
        w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
        return w
    else:
        common = scores.index.intersection(adv.index)
        W = []
        for d in common:
            a = adv.loc[d].reindex(scores.columns).fillna(0.0)
            liq = a.gt(min_adv_usd)
            row = scores.loc[d].loc[liq]
            r = row.rank(ascending=False, method="first")
            picks = r.le(top_n)
            if method=="equal" or vol_ref is None:
                w = (picks.astype(float) / picks.sum()) if picks.sum()>0 else picks.astype(float)
            else:
                w = inverse_vol_weights(vol_ref.loc[d], picks)
            w = w.clip(upper=max_w)
            if w.sum()>0:
                w = w / w.sum()
            W.append(w.reindex(scores.columns).fillna(0.0))
        return pd.DataFrame(W, index=common, columns=scores.columns).fillna(0.0)
