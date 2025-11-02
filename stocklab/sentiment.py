import pandas as pd, numpy as np

def _try_finbert(texts):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        nlp = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, truncation=True)
        out = nlp(texts)
        return out
    except Exception:
        return None

def _vader_scores(texts):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        an = SentimentIntensityAnalyzer()
        return [an.polarity_scores(t)["compound"] for t in texts]
    except Exception:
        return [0.0 for _ in texts]

def score_news(df):
    if df.empty:
        return df.assign(sentiment=0.0)
    texts = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()
    res = _try_finbert(texts)
    if res is not None:
        lab = [r["label"] for r in res]
        val = [r["score"] for r in res]
        mapv = {"positive":1.0,"neutral":0.0,"negative":-1.0}
        s = [mapv.get(l.lower(),0.0)*v for l,v in zip(lab,val)]
    else:
        s = _vader_scores(texts)
    df = df.copy()
    df["sentiment"] = s
    return df

def aggregate_sentiment(df, half_life_days=7):
    if df.empty:
        return pd.Series(dtype=float)
    df = df.dropna(subset=["published"]).copy()
    df["age_days"] = (pd.Timestamp.utcnow() - df["published"]).dt.total_seconds()/86400.0
    w = np.power(0.5, df["age_days"]/half_life_days)
    grp = df.groupby("ticker").apply(lambda g: (g["sentiment"]*w.loc[g.index]).sum()/(w.loc[g.index].sum() if w.loc[g.index].sum()!=0 else 1.0))
    grp.name = "sentiment_score"
    return grp
