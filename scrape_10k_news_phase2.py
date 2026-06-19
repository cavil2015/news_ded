"""
Phase 2 Scraper for 10,000 unique English and Russian news articles.
Uses newspaper3k to extract all article URLs from major news domains,
then downloads them in parallel using trafilatura until we reach exactly 10,000.
"""
import json
import hashlib
import time
import requests
import trafilatura
import newspaper
from datetime import datetime
from pathlib import Path
import sys
import io
import random
import threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

DOMAINS = [
    # English
    "http://cnn.com", "http://bbc.com", "http://nytimes.com", "http://theguardian.com",
    "http://washingtonpost.com", "http://foxnews.com", "http://nbcnews.com", "http://cnbc.com",
    "http://techcrunch.com", "http://theverge.com", "http://wired.com", "http://npr.org",
    "http://usatoday.com", "http://apnews.com", "http://independent.co.uk", "http://telegraph.co.uk",
    "http://bloomberg.com", "http://wsj.com", "http://time.com", "http://newsweek.com",
    "http://vice.com", "http://buzzfeednews.com", "http://huffpost.com", "http://politico.com",
    "http://nypost.com", "http://latimes.com", "http://aljazeera.com", "http://reuters.com",
    "http://chicagotribune.com", "http://boston.com", "http://seattletimes.com",
    "http://forbes.com", "http://cbsnews.com", "http://abcnews.go.com", "http://mashable.com",
    "http://businessinsider.com", "http://theatlantic.com", "http://thedailybeast.com",
    "http://politico.eu", "http://nationalgeographic.com", "http://sciencedaily.com",
    # Russian
    "http://lenta.ru", "http://rbc.ru", "http://ria.ru", "http://tass.ru",
    "http://kommersant.ru", "http://vedomosti.ru", "http://habr.com", "http://vc.ru",
    "http://gazeta.ru", "http://iz.ru", "http://rg.ru", "http://fontanka.ru",
    "http://interfax.ru", "http://mk.ru", "http://kp.ru", "http://aif.ru",
    "http://meduza.io", "http://republic.ru", "http://theins.ru", "http://novayagazeta.ru",
    "http://3dnews.ru", "http://ixbt.com", "http://life.ru", "http://snob.ru",
    "http://rt.com", "http://vesti.ru", "http://ng.ru", "http://rosbalt.ru",
    "http://sports.ru", "http://championat.com", "http://kolesa.ru", "http://drive2.ru",
    "http://cnews.ru", "http://tproger.ru", "http://pikabu.ru", "http://yaplakal.com",
    "http://igromania.ru", "http://kanobu.ru", "http://kino-teatr.ru"
]

TARGET = 15000
OUTPUT_FILE = "data/news_10k_clean.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 10
WORKERS = 20

def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def extract_urls_from_domain(domain):
    try:
        lang = 'ru' if '.ru' in domain or 'meduza' in domain else 'en'
        paper = newspaper.build(domain, language=lang, memoize_articles=False)
        urls = [article.url for article in paper.articles]
        return urls
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
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for art in existing:
            seen_hashes.add(art.get("text_hash", ""))
            seen_urls.add(art.get("url", ""))
        print(f"Loaded {len(existing)} articles from Phase 1.")
    
    if len(existing) >= TARGET:
        print("Target already reached!")
        return

    articles = list(existing)
    
    print(f"\n{'='*60}")
    print("Phase 2: Extracting URLs from homepages using newspaper3k...")
    print(f"{'='*60}")
    
    all_candidates = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(extract_urls_from_domain, domain): domain for domain in DOMAINS}
        for future in as_completed(futures):
            domain = futures[future]
            urls = future.result()
            new_urls = [u for u in urls if u not in seen_urls]
            all_candidates.extend(new_urls)
            print(f"  {domain}: Found {len(urls)} URLs ({len(new_urls)} new)")
            
    print(f"\nTotal new candidate URLs: {len(all_candidates)}")
    
    random.shuffle(all_candidates)
    
    print(f"\n{'='*60}")
    print(f"Phase 2: Downloading articles in PARALLEL (target: {TARGET})...")
    print(f"{'='*60}\n")
    
    errors = 0
    dupes = 0
    start_time = time.time()
    initial_count = len(articles)
    
    lock = threading.Lock()
    
    def process_article(url):
        nonlocal errors, dupes
        
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
            source = urlparse(url).netloc
            
            articles.append({
                "id": article_id,
                "url": url,
                "title": "", # newspaper3k build doesn't fetch title without downloading HTML
                "text": text,
                "source": source,
                "published_at": None,
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
                break

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DONE: {len(articles)} articles saved to {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
