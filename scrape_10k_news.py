"""
Scraper for 10,000 unique English and Russian news articles.
Uses RSS feeds + trafilatura for clean text extraction in PARALLEL.
"""
import json
import hashlib
import time
import feedparser
import requests
import trafilatura
from datetime import datetime
from pathlib import Path
import sys
import io
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------
# RSS FEED LIST: English + Russian sources
# -----------------------------------------------------------------------
RSS_FEEDS = [
    # English - General News
    ("BBC News", "http://feeds.bbci.co.uk/news/rss.xml"),
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Business", "http://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC Technology", "http://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("BBC Science", "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
    ("Reuters Top", "https://feeds.reuters.com/reuters/topNews"),
    ("Reuters World", "https://feeds.reuters.com/reuters/worldNews"),
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Technology", "https://feeds.reuters.com/reuters/technologyNews"),
    ("Reuters Health", "https://feeds.reuters.com/reuters/healthNews"),
    ("AP News", "https://rsshub.app/apnews/topics/apf-topnews"),
    ("The Guardian World", "https://www.theguardian.com/world/rss"),
    ("The Guardian UK", "https://www.theguardian.com/uk/rss"),
    ("The Guardian US", "https://www.theguardian.com/us-news/rss"),
    ("The Guardian Business", "https://www.theguardian.com/business/rss"),
    ("The Guardian Tech", "https://www.theguardian.com/technology/rss"),
    ("The Guardian Science", "https://www.theguardian.com/science/rss"),
    ("The Guardian Environment", "https://www.theguardian.com/environment/rss"),
    ("The Guardian Culture", "https://www.theguardian.com/culture/rss"),
    ("The Guardian Sport", "https://www.theguardian.com/sport/rss"),
    ("NYT Home", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
    ("NYT World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    ("NYT US", "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"),
    ("NYT Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ("NYT Tech", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"),
    ("NYT Science", "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml"),
    ("NYT Health", "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml"),
    ("NYT Sports", "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml"),
    ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
    ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
    ("NPR Politics", "https://feeds.npr.org/1014/rss.xml"),
    ("NPR Business", "https://feeds.npr.org/1006/rss.xml"),
    ("NPR Science", "https://feeds.npr.org/1007/rss.xml"),
    ("NPR Health", "https://feeds.npr.org/1128/rss.xml"),
    ("CNN Top", "http://rss.cnn.com/rss/edition.rss"),
    ("CNN World", "http://rss.cnn.com/rss/edition_world.rss"),
    ("CNN US", "http://rss.cnn.com/rss/edition_us.rss"),
    ("CNN Business", "http://rss.cnn.com/rss/money_news_international.rss"),
    ("CNN Technology", "http://rss.cnn.com/rss/edition_technology.rss"),
    ("CNN Science", "http://rss.cnn.com/rss/edition_space.rss"),
    ("Fox News", "https://moxie.foxnews.com/google-publisher/latest.xml"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Ars Technica", "http://feeds.arstechnica.com/arstechnica/index"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("Wired", "https://www.wired.com/feed/rss"),
    ("Engadget", "https://www.engadget.com/rss.xml"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Al Jazeera Economy", "https://www.aljazeera.com/xml/rss/economy.xml"),
    ("DW News", "https://rss.dw.com/rdf/rss-en-all"),
    ("DW Europe", "https://rss.dw.com/rdf/rss-en-eu"),
    ("France 24", "https://www.france24.com/en/rss"),
    ("France 24 World", "https://www.france24.com/en/tag/world/rss"),
    ("Euronews", "https://www.euronews.com/rss?format=mrss&level=theme&name=news"),
    ("The Independent", "https://www.independent.co.uk/news/rss"),
    ("Telegraph", "https://www.telegraph.co.uk/rss.xml"),
    ("Sky News", "https://feeds.skynews.com/feeds/rss/home.xml"),
    ("Sky News World", "https://feeds.skynews.com/feeds/rss/world.xml"),
    ("CBS News", "https://www.cbsnews.com/latest/rss/main"),
    ("ABC News", "https://abcnews.go.com/abcnews/topstories"),
    ("NBC News", "http://feeds.nbcnews.com/nbcnews/public/news"),
    ("USA Today", "http://rssfeeds.usatoday.com/usatoday-NewsTopStories"),
    ("Washington Post", "https://feeds.washingtonpost.com/rss/homepage"),
    ("The Atlantic", "https://www.theatlantic.com/feed/all/"),
    ("Time", "https://time.com/feed/"),
    ("Newsweek", "https://www.newsweek.com/rss"),
    ("Bloomberg Tech", "https://feeds.bloomberg.com/technology/news.rss"),
    ("Fortune", "https://fortune.com/feed/"),
    ("Business Insider", "https://feeds.businessinsider.com/custom/all"),
    ("Politico", "https://rss.politico.com/politics-news.xml"),
    ("The Hill", "https://thehill.com/feed/"),
    ("Axios", "https://api.axios.com/feed/"),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC World", "https://www.cnbc.com/id/100727362/device/rss/rss.html"),
    ("Financial Times", "https://www.ft.com/rss/home/uk"),
    ("The Economist", "https://www.economist.com/rss/the_world_this_week_rss.xml"),
    ("New Scientist", "https://www.newscientist.com/feed/home/"),
    ("Science Daily", "https://www.sciencedaily.com/rss/top/science.xml"),
    ("Space.com", "https://www.space.com/feeds/all"),
    ("NASA", "https://www.nasa.gov/rss/dyn/breaking_news.rss"),
    ("Nature", "https://www.nature.com/nature.rss"),
    ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
    ("Hacker News", "https://hnrss.org/frontpage"),
    ("Sports Illustrated", "https://www.si.com/rss/si_topstories.rss"),
    ("ESPN", "https://www.espn.com/espn/rss/news"),
    ("Health.com", "https://www.health.com/syndication/rss/all"),
    ("WebMD", "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC"),
    ("Harvard Health", "https://www.health.harvard.edu/blog/feed"),
    ("National Geographic", "https://www.nationalgeographic.com/rss/"),
    ("Popular Science", "https://www.popsci.com/rss.xml"),
    # Russian sources
    ("РИА Новости", "https://ria.ru/export/rss2/archive/index.xml"),
    ("РИА Мир", "https://ria.ru/export/rss2/world/index.xml"),
    ("РИА Экономика", "https://ria.ru/export/rss2/economy/index.xml"),
    ("РИА Политика", "https://ria.ru/export/rss2/politics/index.xml"),
    ("РИА Наука", "https://ria.ru/export/rss2/science/index.xml"),
    ("РИА Спорт", "https://ria.ru/export/rss2/sport/index.xml"),
    ("РИА Культура", "https://ria.ru/export/rss2/culture/index.xml"),
    ("ТАСС", "https://tass.ru/rss/v2.xml"),
    ("Интерфакс", "https://www.interfax.ru/rss.asp"),
    ("Коммерсантъ", "https://www.kommersant.ru/RSS/main.xml"),
    ("Коммерсантъ Бизнес", "https://www.kommersant.ru/RSS/corp.xml"),
    ("Коммерсантъ Политика", "https://www.kommersant.ru/RSS/news.xml"),
    ("Ведомости", "https://www.vedomosti.ru/rss/news"),
    ("Известия", "https://iz.ru/xml/rss/all.xml"),
    ("Российская газета", "https://rg.ru/xml/index.xml"),
    ("Fontanka.ru", "https://www.fontanka.ru/fontanka.rss"),
    ("Meduza", "https://meduza.io/rss/all"),
    ("Republic", "https://republic.ru/rss"),
    ("Новая газета", "https://novayagazeta.ru/rss/novosti.xml"),
    ("The Insider", "https://theins.ru/feed"),
    ("iXBT", "https://www.ixbt.com/export/news.rss"),
    ("3DNews", "https://3dnews.ru/news/rss/"),
    ("Хабр", "https://habr.com/ru/rss/hubs/all/"),
    ("РБК", "https://rbc.ru/v10/ajax/get-news-feed/project/rbcnews/lastDate/0/limit/100"),
    ("Газета.ру", "https://www.gazeta.ru/export/rss/lenta.xml"),
    ("Life", "https://life.ru/xml/atom.xml"),
]

TARGET = 10000
OUTPUT_FILE = "data/news_10k_clean.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 10
WORKERS = 20


def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def fetch_rss_urls(feed_name, feed_url):
    try:
        d = feedparser.parse(feed_url, request_headers=HEADERS)
        articles = []
        for entry in d.entries:
            url = entry.get("link", "")
            title = entry.get("title", "")
            date = entry.get("published", entry.get("updated", ""))
            if url:
                articles.append((url, title, date, feed_name))
        return articles
    except Exception as e:
        return []


def fetch_article_text(url):
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        if resp.status_code != 200:
            return None
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return text
    except Exception:
        return None


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    
    Path("data").mkdir(exist_ok=True)
    
    existing = []
    seen_hashes = set()
    seen_urls = set()
    
    if Path(OUTPUT_FILE).exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for art in existing:
                seen_hashes.add(art.get("text_hash", ""))
                seen_urls.add(art.get("url", ""))
            print(f"Resuming from {len(existing)} articles already saved.")
        except Exception:
            print("Failed to load existing json. Starting fresh.")
    
    articles = list(existing)
    
    print(f"\n{'='*60}")
    print("Phase 1: Collecting URLs from RSS feeds...")
    print(f"{'='*60}")
    
    all_candidates = []
    
    # Phase 1 can also be parallelized
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_rss_urls, name, url): name for name, url in RSS_FEEDS}
        for future in as_completed(futures):
            entries = future.result()
            new_entries = [(u, t, d, s) for u, t, d, s in entries if u not in seen_urls]
            all_candidates.extend(new_entries)
    
    print(f"\nTotal new candidates: {len(all_candidates)}")
    
    # SHUFFLE candidates to distribute load across domains evenly
    random.shuffle(all_candidates)
    
    print(f"\n{'='*60}")
    print(f"Phase 2: Downloading articles in PARALLEL (target: {TARGET})...")
    print(f"{'='*60}\n")
    
    errors = 0
    dupes = 0
    start_time = time.time()
    initial_count = len(articles)
    
    lock = threading.Lock()
    
    def process_article(candidate):
        nonlocal errors, dupes
        url, title, date, source = candidate
        
        with lock:
            if len(articles) >= TARGET: return
            if url in seen_urls: return
            seen_urls.add(url)
            
        text = fetch_article_text(url)
        
        if not text or len(text.strip()) < 100:
            with lock: errors += 1
            return
            
        h = text_hash(text)
        
        with lock:
            if h in seen_hashes:
                dupes += 1
                return
            seen_hashes.add(h)
            
            if len(articles) >= TARGET: return
            
            article_id = len(articles) + 1
            articles.append({
                "id": article_id,
                "url": url,
                "title": title,
                "text": text,
                "source": source,
                "published_at": date,
                "text_hash": h,
                "true_cluster_id": None
            })
            
            if len(articles) % 25 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(articles, f, ensure_ascii=False, indent=2)
                
                elapsed = time.time() - start_time
                articles_per_sec = (len(articles) - initial_count) / elapsed if elapsed > 0 else 0
                rem_articles = TARGET - len(articles)
                eta_sec = rem_articles / articles_per_sec if articles_per_sec > 0 else 0
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {len(articles)}/{TARGET} | "
                      f"Errors: {errors} | Dupes: {dupes} | "
                      f"Speed: {articles_per_sec:.2f} it/s | ETA: {int(eta_sec//60)}m {int(eta_sec%60)}s", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = []
        for c in all_candidates:
            if len(articles) >= TARGET:
                break
            futures.append(executor.submit(process_article, c))
            
        for future in as_completed(futures):
            if len(articles) >= TARGET:
                # Cancel remaining
                break

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DONE: {len(articles)} articles saved to {OUTPUT_FILE}")
    print(f"  Errors (empty/short): {errors}")
    print(f"  Duplicates skipped:   {dupes}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
