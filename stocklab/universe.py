import time
import pandas as pd
import requests
import yfinance as yf

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

def _fetch_html(url, retries=4, backoff=1.8, timeout=20):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=timeout)
            if r.status_code == 200 and len(r.text) > 10000:
                return r.text
        except Exception as e:
            last_err = e
        time.sleep(backoff**i)
    if last_err:
        raise last_err
    raise RuntimeError(f"failed to fetch {url}")

def _read_first_table_with(cols_required, html):
    tables = pd.read_html(html)
    for t in tables:
        cols = {c.strip().lower() for c in t.columns}
        need = {c.lower() for c in cols_required}
        if need.issubset(cols):
            return t
    return None

def sp500_table():
    urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies?useskin=vector"
    ]
    last = None
    for u in urls:
        html = _fetch_html(u)
        t = _read_first_table_with({"Symbol", "GICS Sector"}, html)
        if t is not None:
            df = t.copy()
            if "Symbol" not in df.columns and "Ticker symbol" in df.columns:
                df = df.rename(columns={"Ticker symbol": "Symbol"})
            df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
            return df
        last = u
    raise RuntimeError(f"no table with expected columns from {last}")

def sp500_tickers():
    df = sp500_table()
    return df["Symbol"].tolist()

def sp500_sector_map():
    df = sp500_table()
    return df.set_index("Symbol")["GICS Sector"].to_dict()

def nasdaq100_tickers():
    urls = [
        "https://en.wikipedia.org/wiki/NASDAQ-100",
        "https://en.m.wikipedia.org/wiki/NASDAQ-100"
    ]
    for u in urls:
        html = _fetch_html(u)
        t1 = _read_first_table_with({"Ticker"}, html)
        t2 = _read_first_table_with({"Symbol"}, html)
        t = t1 if t1 is not None else t2
        if t is not None:
            col = "Ticker" if "Ticker" in t.columns else "Symbol"
            return t[col].astype(str).str.replace(".", "-", regex=False).tolist()
    return []

def build_universe(source="sp500"):
    if source == "sp500":
        return sp500_tickers()
    if source == "nasdaq100":
        return nasdaq100_tickers()
    return []

def filter_liquid(tickers, min_price=5, min_adv_usd=3_000_000, lookback_days=60):
    ohlc = yf.download(tickers, period=f"{lookback_days}d", auto_adjust=False, threads=True, group_by="column")
    px = ohlc["Close"].tail(1).squeeze()
    adv = (ohlc["Close"] * ohlc["Volume"]).rolling(20).mean().tail(1).squeeze()
    if isinstance(px, pd.Series):
        mask = px.ge(min_price) & adv.ge(min_adv_usd)
        return list(px.index[mask])
    return tickers

def write_universe_csv(tickers, path="data/universe.csv"):
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)