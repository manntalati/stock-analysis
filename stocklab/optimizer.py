import itertools, pandas as pd
from .score import composite
from .portfolio import topn_weights
from .backtest import run

def grid_search(features, prices, adv, params, start=None, end=None):
    keys = list(params.keys())
    results = []
    idx = features.index
    if start or end:
        idx = idx[(idx>=start) & (idx<=end)] if start and end else idx[(idx>=start)] if start else idx[(idx<=end)]
    for vals in itertools.product(*params.values()):
        cfg = dict(zip(keys, vals))
        s, _ = composite(features.loc[idx], None, cfg.get("w_mom",0.5), cfg.get("w_trend",0.25), cfg.get("w_risk",0.15), cfg.get("w_pred",0.1))
        W = []
        for d in s.index:
            arow = adv.reindex(columns=s.columns).loc[d] if d in adv.index else pd.Series(0.0, index=s.columns)
            w = topn_weights(s.loc[d].to_frame().T, arow, top_n=cfg.get("top_n",20), max_w=cfg.get("max_w",0.08), min_adv_usd=cfg.get("min_adv_usd",1_000_000))
            W.append(w.iloc[0])
        W = pd.DataFrame(W, index=s.index, columns=s.columns).fillna(0.0)
        res = run(prices.reindex(index=W.index, columns=W.columns), W, cfg.get("tx_cost_bps",5))
        results.append({**cfg, "Sharpe":res["sharpe"], "CAGR":res["cagr"], "Vol":res["vol"], "MaxDD":res["maxdd"]})
    return pd.DataFrame(results)
