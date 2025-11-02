import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

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

def fit_predict(features, prices, h=5, n_splits=5):
    F = _unflatten_features(features)
    X = F.stack(level=1).sort_index()
    y = forward_return(prices, h).stack().rename("y").sort_index()
    df = X.join(y).dropna()
    Xv = df.drop(columns=["y"])
    yv = df["y"]
    pipe = Pipeline([("sc", StandardScaler(with_mean=False)), ("rr", Ridge())])
    params = {"rr__alpha":[0.1,0.5,1.0,2.0,5.0]}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    g = GridSearchCV(pipe, params, cv=tscv, n_jobs=None)
    preds = pd.Series(index=yv.index, dtype=float)
    for tr, te in tscv.split(Xv):
        g.fit(Xv.iloc[tr], yv.iloc[tr])
        preds.iloc[te] = g.predict(Xv.iloc[te])
    P = preds.unstack()
    return P
