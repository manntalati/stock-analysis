import pandas as pd, datetime as dt, os

def _coerce_datetime_index(s):
    idx = s.index
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(0)
    try:
        idx = pd.to_datetime(idx)
    except Exception:
        idx = pd.to_datetime([i[0] if isinstance(i, tuple) else i for i in s.index])
    s = s.copy()
    s.index = idx
    s = s.sort_index()
    return s

def month_table(rets):
    r = _coerce_datetime_index(rets)
    df = r.to_frame("r")
    df["Y"] = df.index.year
    df["M"] = df.index.month
    t = df.pivot_table(index="Y", columns="M", values="r", aggfunc=lambda x: (1+x).prod()-1).fillna(0.0)
    return t

def to_html(metrics, curve, rets, out_path):
    curve = _coerce_datetime_index(curve)
    rets = _coerce_datetime_index(rets)
    mt = month_table(rets)
    h = []
    h.append("<html><head><meta charset='utf-8'><title>stocklab report</title><style>table,th,td{border:1px solid #ccc;border-collapse:collapse;padding:4px} body{font-family:Arial, sans-serif;padding:16px}</style></head><body>")
    h.append("<h2>Summary</h2><table>")
    for k,v in metrics.items():
        try:
            h.append(f"<tr><td>{k}</td><td>{float(v):.6f}</td></tr>")
        except Exception:
            h.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    h.append("</table>")
    h.append("<h2>Monthly Returns</h2>")
    h.append(mt.to_html())
    h.append("<h2>Equity Curve (CSV)</h2>")
    h.append("<p>Saved as data/reports/equity_curve.parquet and equity_curve.csv</p>")
    h.append("</body></html>")
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("".join(h))