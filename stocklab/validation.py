import pandas as pd
import numpy as np

def check_nan_rates(df, max_rate=0.05):
    r = df.isna().mean().max()
    return r<=max_rate, r

def assert_no_lookahead(features, prices):
    last_feat = features.index.max()
    last_price = prices.index.max()
    return last_feat<=last_price

def align_index_left(a, b):
    idx = a.index.intersection(b.index)
    return a.reindex(idx), b.reindex(idx)

def basic_integrity_report(ohlc, prices, features):
    ok1 = not ohlc.isna().any().any()
    ok2 = features.index.min()>=prices.index.min()
    ok3 = features.index.max()<=prices.index.max()
    return {"ohlc_no_nans":ok1,"features_in_range":ok2 and ok3}
