import argparse, pandas as pd, yaml, logging
from pathlib import Path
import yfinance as yf
from .features import build_features
from .predict import fit_predict
from .score import composite
from .portfolio import topn_weights
from .backtest import run
from .metrics import perf_summary
from .io_data import fetch_ohlc, fetch_prices, save_parquet, load_parquet, compute_adv_usd
from .universe import build_universe, filter_liquid, write_universe_csv, sp500_sector_map
from .report import save_top_picks, save_summary
from .report_html import to_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def _dt_index(idx):
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(0)
    try:
        return pd.to_datetime(idx)
    except Exception:
        vals = [x[0] if isinstance(x, tuple) else x for x in list(idx)]
        return pd.to_datetime(vals)

def _dt(s):
    s = s.copy()
    s.index = _dt_index(s.index)
    s = s.sort_index()
    return s

def load_config():
    p = Path("config/universe.yaml")
    if p.exists():
        with open(p, "r") as f:
            return yaml.safe_load(f)
    return {"universe":{"source":"sp500","min_price":5,"min_adv_usd":3000000},"params":{"top_n":20,"max_pos_w":0.08,"min_adv_usd":1000000,"tx_cost_bps":5},"scores":{"w_mom":0.5,"w_trend":0.25,"w_risk":0.15,"w_pred":0.1},"report":{"benchmark":"SPY"}}

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("cmd")
    pa.add_argument("--start")
    pa.add_argument("--end")
    pa.add_argument("--date")
    pa.add_argument("--top", type=int)
    pa.add_argument("--source")
    args = pa.parse_args()
    cfg = load_config()

    if args.cmd == "build-universe":
        src = args.source or cfg["universe"].get("source","sp500")
        tickers = build_universe(src)
        tickers = filter_liquid(tickers, cfg["universe"]["min_price"], cfg["universe"]["min_adv_usd"])
        write_universe_csv(tickers)
        try:
            pd.Series(sp500_sector_map()).to_csv("data/sector_map.csv")
        except Exception:
            pass
        print(len(tickers))

    if args.cmd == "fetch":
        tickers = pd.read_csv("data/universe.csv")["ticker"].tolist()
        ohlc = fetch_ohlc(tickers, start=args.start, end=args.end, auto_adjust=False)
        prices = ohlc["Close"]
        save_parquet("data/raw/ohlc.parquet", ohlc)
        save_parquet("data/raw/prices.parquet", prices)
        adv = compute_adv_usd(ohlc, 20)
        save_parquet("data/raw/adv.parquet", adv)

    if args.cmd == "features":
        ohlc = load_parquet("data/raw/ohlc.parquet")
        feats = build_features(ohlc)
        save_parquet("data/features/features.parquet", feats)

    if args.cmd == "predict":
        px = load_parquet("data/raw/prices.parquet")
        feats = load_parquet("data/features/features.parquet")
        preds = fit_predict(feats, px)
        save_parquet("data/features/preds.parquet", preds)

    if args.cmd == "score":
        feats = load_parquet("data/features/features.parquet")
        try:
            preds = load_parquet("data/features/preds.parquet")
        except Exception:
            preds = None
        s, parts = composite(feats, preds, cfg["scores"]["w_mom"], cfg["scores"]["w_trend"], cfg["scores"]["w_risk"], cfg["scores"]["w_pred"])
        save_parquet("data/signals/scores.parquet", s)
        for k,v in parts.items():
            save_parquet(f"data/signals/part_{k}.parquet", v)

    if args.cmd == "backtest":
        px = _dt(load_parquet("data/raw/prices.parquet"))
        s  = _dt(load_parquet("data/signals/scores.parquet"))
        adv = _dt(load_parquet("data/raw/adv.parquet"))
        start = pd.to_datetime(args.start) if args.start else None
        end   = pd.to_datetime(args.end) if args.end else None
        if start is not None:
            s = s.loc[s.index >= start]
        if end is not None:
            s = s.loc[s.index <= end]
        W = []
        for d in s.index:
            arow = adv.reindex(columns=s.columns).loc[d] if d in adv.index else pd.Series(0.0, index=s.columns)
            w = topn_weights(s.loc[d].to_frame().T, arow, top_n=cfg["params"]["top_n"], max_w=cfg["params"]["max_pos_w"], min_adv_usd=cfg["params"]["min_adv_usd"])
            W.append(w.iloc[0])
        W = pd.DataFrame(W, index=s.index, columns=s.columns).fillna(0.0)
        px2 = px.reindex(index=W.index, columns=W.columns)
        res = run(px2, W, cfg["params"]["tx_cost_bps"])
        save_parquet("data/reports/equity_curve.parquet", res["curve"].to_frame("equity"))
        res["curve"].to_frame("equity").to_csv("data/reports/equity_curve.csv")
        save_summary(res, "data/reports/summary.csv")

    if args.cmd == "picks":
        s = _dt(load_parquet("data/signals/scores.parquet"))
        adv = _dt(load_parquet("data/raw/adv.parquet"))
        date = pd.to_datetime(args.date) if args.date else pd.Timestamp.today().normalize()
        if date not in s.index:
            date = s.index.max()

        s_at = s.loc[date]
        if isinstance(s_at, pd.DataFrame):
            srow = s_at.iloc[-1]
        else:
            srow = s_at

        adv_all = adv.reindex(columns=s.columns)
        if date in adv_all.index:
            a_at = adv_all.loc[date]
            if isinstance(a_at, pd.DataFrame):
                arow = a_at.iloc[-1]
            else:
                arow = a_at
        else:
            arow = pd.Series(0.0, index=s.columns)

        top = args.top or cfg["params"]["top_n"]
        scores_df = srow.to_frame().T
        w = topn_weights(
            scores_df,
            arow,
            top_n=top,
            max_w=cfg["params"]["max_pos_w"],
            min_adv_usd=cfg["params"]["min_adv_usd"]
        )
        Path("data/signals").mkdir(parents=True, exist_ok=True)
        out = f"data/signals/top_picks_{date.date()}.csv"
        save_top_picks(w.iloc[0], srow, out)
        print(out)

    if args.cmd == "report-html":
        df = load_parquet("data/reports/equity_curve.parquet")
        curve = df["equity"] if "equity" in df.columns else df.squeeze()
        curve.index = _dt_index(curve.index)
        rets = curve.pct_change(fill_method=None).fillna(0.0)
        met = perf_summary(rets)
        to_html(met, curve, rets, "data/reports/report.html")
        print("data/reports/report.html")

if __name__ == "__main__":
    main()