import feedparser
import pandas as pd

feeds = {
    # FAKE - Indian fact check sites
    "altnews":      "https://www.altnews.in/feed/",
    "boomlive":     "https://www.boomlive.in/fact-check/feed",
    "newschecker":  "https://newschecker.in/feed",
    "factchecker":  "https://factchecker.in/feed/",
    
    # REAL - Indian credible news  
    "thehindu":     "https://www.thehindu.com/news/national/feeder/default.rss",
    "ndtv":         "https://feeds.feedburner.com/ndtvnews-india-news",
    "indianexpress":"https://indianexpress.com/section/india/feed/",
    "pib":          "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
}

FAKE_SOURCES = ["altnews", "boomlive", "newschecker", "factchecker"]

data = []
for source, url in feeds.items():
    try:
        feed = feedparser.parse(url)
        label = 0 if source in FAKE_SOURCES else 1
        count = 0
        for entry in feed.entries:
            text = entry.get('title','') + " " + entry.get('summary','')
            if len(text.split()) > 15:
                data.append({
                    "content": text,
                    "label": label,
                    "source": source
                })
                count += 1
        print(f"✅ {source}: {count} articles (label={label})")
    except Exception as e:
        print(f"❌ {source}: {e}")

df = pd.DataFrame(data)
print(f"\n=== Results ===")
print(f"Total:  {len(df)}")
print(f"Fake:   {(df['label']==0).sum()}")
print(f"Real:   {(df['label']==1).sum()}")
df.to_csv("indian_news.csv", index=False)
print("✅ Saved to indian_news.csv")