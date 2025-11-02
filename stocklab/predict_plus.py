import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

def forward_return(px, h=5):
    return px.pct_change(h).shift(-h)

def _unflatten_features(features):
    cols = features.columns
    parts = [c.split("::",1) for c in cols]
    feats = [p[0] for p in parts]
    tickers = [p[1] for p in parts]
    mi = pd.MultiIndex.from_arrays([feats, tickers], names=["feat","ticker"])
    F = features.copy()
    F.columns = mi
    return F

def fit_predict_ensemble(features, prices, h=5, sentiment=None, n_splits=5):
    F = _unflatten_features(features)
    X = F.stack(level=1).sort_index()
    y = forward_return(prices, h).stack().rename("y").sort_index()
    df = X.join(y).dropna()
    if sentiment is not None and isinstance(sentiment, pd.Series):
        sent = sentiment.reindex(df.index.get_level_values(1)).values
        df = df.assign(sentiment=sent)
    Xv = df.drop(columns=["y"])
    yv = df["y"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = pd.Series(index=yv.index, dtype=float)
    for tr, te in tscv.split(Xv):
        Xtr, ytr = Xv.iloc[tr], yv.iloc[tr]
        Xte = Xv.iloc[te]
        lin = Pipeline([("sc", StandardScaler(with_mean=False)), ("rr", Ridge(alpha=0.5))])
        gbr = GradientBoostingRegressor(random_state=42)
        lin.fit(Xtr, ytr)
        gbr.fit(Xtr, ytr)
        p1 = lin.predict(Xte)
        p2 = gbr.predict(Xte)
        preds.iloc[te] = 0.6*p1 + 0.4*p2
    P = preds.unstack()
    return P
