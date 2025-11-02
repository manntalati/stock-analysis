import pandas as pd

def save_top_picks(weights_row, scores_row, path):
    df = pd.DataFrame({"weight": weights_row, "score": scores_row}).sort_values("weight", ascending=False)
    df.to_csv(path)

def save_summary(res, path_csv):
    pd.Series({"Sharpe":res.get("sharpe",0.0),"Sortino":res.get("sortino",0.0),"CAGR":res.get("cagr",0.0),"Vol":res.get("vol",0.0),"MaxDD":res.get("maxdd",0.0),"Turnover":res.get("turnover",0.0)}).to_csv(path_csv)
