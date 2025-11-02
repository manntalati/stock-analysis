import datetime as dt, pandas as pd, requests, time

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

def fetch_gnews(api_key, query, lang="en", max_items=10):
    if not api_key:
        return []
    url = "https://gnews.io/api/v4/search"
    params = {"q": query, "lang": lang, "max": max_items, "in":"title,description","sortby":"publishedAt","expand":"content","apikey":api_key}
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=20)
    if r.status_code!=200:
        return []
    js = r.json()
    out = []
    for a in js.get("articles", []):
        out.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "source": a.get("source",{}).get("name"),
            "published": a.get("publishedAt"),
            "description": a.get("description")
        })
    return out

def headlines_for_tickers(api_key, tickers, max_per=5):
    rows = []
    for t in tickers:
        qs = f"{t} stock OR {t} earnings"
        items = fetch_gnews(api_key, qs, max_items=max_per)
        for it in items:
            it["ticker"] = t
            rows.append(it)
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame(columns=["ticker","title","url","source","published","description"])
    df = pd.DataFrame(rows)
    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    return df.sort_values("published", ascending=False)
