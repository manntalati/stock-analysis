import os, sys, json, subprocess, datetime as dt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from stocklab.sentiment import aggregate_sentiment
from stocklab.news import headlines_for_tickers
from stocklab.forecast import simulate_next_week

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
SIG_PATH = os.path.join(DATA_DIR, "signals", "scores.parquet")
PRED_PATH = os.path.join(DATA_DIR, "features", "preds.parquet")
EQ_PATH = os.path.join(DATA_DIR, "reports", "equity_curve.parquet")
SUM_PATH = os.path.join(DATA_DIR, "reports", "summary.csv")
PRC_PATH = os.path.join(DATA_DIR, "raw", "prices.parquet")
WL_DIR = os.path.join(DATA_DIR, "watchlists")
WL_FILE = os.path.join(WL_DIR, "default.json")

def _dt_index(idx):
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(0)
    try:
        return pd.to_datetime(idx)
    except Exception:
        vals = [x[0] if isinstance(x, tuple) else x for x in list(idx)]
        return pd.to_datetime(vals)

@st.cache_data(show_spinner=False)
def load_scores():
    s = pd.read_parquet(SIG_PATH)
    s.index = _dt_index(s.index)
    s = s.sort_index()
    return s

@st.cache_data(show_spinner=False)
def load_preds():
    if os.path.exists(PRED_PATH):
        p = pd.read_parquet(PRED_PATH)
        p.index = _dt_index(p.index)
        p = p.sort_index()
        return p
    return None

@st.cache_data(show_spinner=False)
def load_prices():
    px = pd.read_parquet(PRC_PATH)
    px.index = _dt_index(px.index)
    px = px.sort_index()
    return px

@st.cache_data(show_spinner=False)
def load_equity():
    if os.path.exists(EQ_PATH):
        df = pd.read_parquet(EQ_PATH)
        s = df["equity"] if "equity" in df.columns else df.squeeze()
        s.index = _dt_index(s.index)
        s = s.sort_index()
        return s
    return None

@st.cache_data(show_spinner=False)
def load_summary():
    if os.path.exists(SUM_PATH):
        return pd.read_csv(SUM_PATH, header=None, index_col=0)[1].astype(float)
    return None

def ensure_watchlist_dir():
    os.makedirs(WL_DIR, exist_ok=True)
    if not os.path.exists(WL_FILE):
        with open(WL_FILE,"w") as f:
            json.dump({"name":"Default","tickers":["AAPL","MSFT","NVDA"]}, f)

def read_watchlist():
    ensure_watchlist_dir()
    with open(WL_FILE,"r") as f:
        return json.load(f)

def write_watchlist(d):
    ensure_watchlist_dir()
    with open(WL_FILE,"w") as f:
        json.dump(d, f)

def sidebar_controls(scores):
    st.sidebar.header("Controls")
    dates = list(scores.index.unique())
    d = st.sidebar.selectbox("Date", dates, index=len(dates)-1, format_func=lambda x: x.strftime("%Y-%m-%d"))
    top_n = st.sidebar.slider("Top N", 5, 50, 20, 1)
    gnews_api = st.sidebar.text_input("GNews API key (optional for news/sentiment)", type="password")
    return d, top_n, gnews_api

def topn_table(scores, preds, d, top_n):
    row = scores.loc[d]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    df = row.sort_values(ascending=False).head(top_n).to_frame("score")
    df["score"] = df["score"].round(3)
    if preds is not None and d in preds.index:
        pr = preds.loc[d]
        if isinstance(pr, pd.DataFrame):
            pr = pr.iloc[-1]
        df["pred_5d_%"] = (100*pr.reindex(df.index)).round(2)
    return df

def render_overview(scores, preds, equity, summary):
    d, top_n, gnews_api = sidebar_controls(scores)
    df = topn_table(scores, preds, d, top_n)
    st.subheader(f"Top {top_n} Picks for {d.date()}")
    st.dataframe(df, use_container_width=True)
    if equity is not None:
        st.plotly_chart(px.line(equity, title="Equity Curve"), use_container_width=True)
    if summary is not None:
        st.write("Performance Summary")
        st.table(summary.to_frame("value").style.format("{:.4f}"))
    with st.expander("News & Sentiment for Top Picks"):
        tickers = df.index.tolist()
        news = headlines_for_tickers(gnews_api, tickers, max_per=5) if gnews_api else pd.DataFrame(columns=["ticker","title","url","source","published","description"])
        if news.empty:
            st.info("Provide a GNews API key in the sidebar to fetch headlines.")
        else:
            from stocklab.sentiment import score_news, aggregate_sentiment
            scored = score_news(news)
            agg = aggregate_sentiment(scored)
            st.write("Ticker sentiment (time-decayed):")
            st.dataframe(agg.to_frame("sentiment").round(3))
            st.write("Latest headlines:")
            for t in tickers:
                st.markdown(f"**{t}**")
                sub = scored[scored["ticker"]==t].head(5)
                for _, r in sub.iterrows():
                    st.write(f"- [{r['title']}]({r['url']}) ({r.get('source','')}, {str(r.get('published',''))[:10]}) — sentiment: {r['sentiment']:.2f}")
    return d, df

def page_watchlist(scores, preds, prices):
    wl = read_watchlist()
    st.subheader("Watchlist")
    df = pd.DataFrame({"ticker": wl.get("tickers", [])})
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True)
    new_list = [t.strip().upper() for t in df["ticker"].dropna().tolist() if t.strip()!=""]
    if st.button("Save Watchlist"):
        write_watchlist({"name": wl.get("name","Default"), "tickers": new_list})
        st.success("Saved")
    if len(new_list)>0:
        d = scores.index.max()
        srow = scores.loc[d]
        if isinstance(srow, pd.DataFrame):
            srow = srow.iloc[-1]
        sub = srow.reindex(new_list).dropna().sort_values(ascending=False).to_frame("score")
        if preds is not None and d in preds.index:
            pr = preds.loc[d]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[-1]
            sub["pred_5d_%"] = (100*pr.reindex(sub.index)).round(2)
        st.markdown(f"Signals for {d.date()}")
        st.dataframe(sub, use_container_width=True)
        pick = st.selectbox("Detail", new_list)
        if pick and pick in prices.columns:
            p = prices[pick].dropna()
            fig = px.line(p.tail(750), title=f"{pick} Price (last ~3y)")
            st.plotly_chart(fig, use_container_width=True)
            if preds is not None and d in preds.index:
                try:
                    pred5 = float(pr[pick])
                    from stocklab.forecast import simulate_next_week
                    stats, sims = simulate_next_week(p, pred5)
                    if stats is not None:
                        g = go.Figure()
                        g.add_trace(go.Scatter(x=stats.index, y=stats["p50"], mode="lines", name="p50"))
                        g.add_trace(go.Scatter(x=stats.index, y=stats["p10"], mode="lines", name="p10"))
                        g.add_trace(go.Scatter(x=stats.index, y=stats["p90"], mode="lines", name="p90"))
                        g.update_layout(title=f"{pick} 5-day forecast")
                        st.plotly_chart(g, use_container_width=True)
                except Exception:
                    pass

def page_run_pipeline():
    st.subheader("Run Pipeline Now")
    st.write("Runs: build-universe, fetch, features, predict, score, backtest, report-html, picks(today)")
    if st.button("Run now"):
        cmds = [
            ["python","-m","stocklab.cli","build-universe","--source","sp500"],
            ["python","-m","stocklab.cli","fetch","--start","2015-01-01"],
            ["python","-m","stocklab.cli","features"],
            ["python","-m","stocklab.cli","predict"],
            ["python","-m","stocklab.cli","score"],
            ["python","-m","stocklab.cli","backtest","--start","2018-01-01","--end","2024-12-31"],
            ["python","-m","stocklab.cli","report-html"],
        ]
        for c in cmds:
            st.write(" ".join(c))
            r = subprocess.run(c, capture_output=True, text=True)
            st.code(r.stdout + "\n" + r.stderr)
        today = dt.date.today().isoformat()
        r = subprocess.run(["python","-m","stocklab.cli","picks","--date",today,"--top","20"], capture_output=True, text=True)
        st.code(r.stdout + "\n" + r.stderr)
        st.success("Done. Refresh the page.")

def main():
    st.set_page_config(page_title="stocklab dashboard+", layout="wide")
    st.title("stocklab dashboard+")
    tab1, tab2, tab3 = st.tabs(["Overview","Watchlist","Run Pipeline"])
    scores = load_scores()
    preds = load_preds()
    equity = load_equity()
    summary = load_summary()
    prices = load_prices()
    with tab1:
        render_overview(scores, preds, equity, summary)
    with tab2:
        page_watchlist(scores, preds, prices)
    with tab3:
        page_run_pipeline()

if __name__ == "__main__":
    main()
