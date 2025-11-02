import numpy as np, pandas as pd

def simulate_next_week(prices_series, pred_pct_5d, n_sims=500):
    s = prices_series.dropna()
    if len(s)<60:
        return None
    last = s.iloc[-1]
    rets = s.pct_change().dropna()
    vol = rets.tail(63).std()
    mu = pred_pct_5d/5.0
    sims = np.zeros((5, n_sims))
    for j in range(n_sims):
        path = []
        p = last
        for i in range(5):
            r = np.random.normal(mu, vol)
            p = p*(1+r)
            path.append(p)
        sims[:, j] = path
    idx = pd.date_range(s.index[-1], periods=6, freq="B")[1:]
    df = pd.DataFrame(sims, index=idx)
    stats = df.quantile([0.1,0.5,0.9], axis=1).T
    stats.index = idx
    stats.columns = ["p10","p50","p90"]
    return stats, df
