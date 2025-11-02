import pandas as pd
import yfinance as yf
from pathlib import Path

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def save_parquet(path, df):
    ensure_dir(path)
    df.to_parquet(path)

def load_parquet(path):
    return pd.read_parquet(path)

def fetch_ohlc(tickers, start=None, end=None, auto_adjust=False, threads=True):
    df = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust, threads=threads, group_by='column')
    return df

def fetch_prices(tickers, start=None, end=None):
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, threads=True)['Close']
    return px

def compute_adv_usd(ohlc, w=20):
    px = ohlc['Close']
    vol = ohlc['Volume']
    adv = (px*vol).rolling(w).mean()
    return adv
