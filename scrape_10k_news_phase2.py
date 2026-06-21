"""
Phase 2 Scraper for unique English and Russian news articles.
Uses newspaper3k to extract all article URLs from major news domains,
then downloads them in parallel using trafilatura.
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
import logging
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("NewsScraper")

class ParallelNewsScraper:
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

    def __init__(self, target: int = 15000, output_file: str = "data/news_10k_clean.json", workers: int = 20):
        self.target = target
        self.output_file = output_file
        self.workers = workers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.request_timeout = 10
        
        self.articles = []
        self.seen_hashes = set()
        self.seen_urls = set()
        
        self.errors = 0
        self.dupes = 0
        self.start_time = 0
        self.initial_count = 0
        
        self.lock = threading.Lock()
        
    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def _extract_urls_from_domain(self, domain: str) -> list:
        try:
            lang = 'ru' if '.ru' in domain or 'meduza' in domain else 'en'
            paper = newspaper.build(domain, language=lang, memoize_articles=False)
            return [article.url for article in paper.articles]
        except Exception:
            return []

    def _fetch_article_text(self, url: str) -> str | None:
        try:
            resp = requests.get(url, timeout=self.request_timeout, headers=self.headers)
            if resp.status_code != 200:
                return None
            return trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
        except Exception:
            return None

    def _load_existing_data(self):
        Path("data").mkdir(exist_ok=True)
        if Path(self.output_file).exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                self.articles = json.load(f)
            for art in self.articles:
                self.seen_hashes.add(art.get("text_hash", ""))
                self.seen_urls.add(art.get("url", ""))
            logger.info(f"Loaded {len(self.articles)} existing articles.")

    def _save_progress(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)

    def _process_article(self, url: str):
        with self.lock:
            if len(self.articles) >= self.target: return
            if url in self.seen_urls: return
            self.seen_urls.add(url)
            
        text = self._fetch_article_text(url)
        
        if not text or len(text.strip()) < 100:
            with self.lock:
                self.errors += 1
            return
            
        h = self._text_hash(text)
        
        with self.lock:
            if h in self.seen_hashes:
                self.dupes += 1
                return
            self.seen_hashes.add(h)
            
            if len(self.articles) >= self.target: return
            
            article_id = len(self.articles) + 1
            source = urlparse(url).netloc
            
            self.articles.append({
                "id": article_id,
                "url": url,
                "title": "",
                "text": text,
                "source": source,
                "published_at": None,
                "text_hash": h,
                "true_cluster_id": None
            })
            
            if len(self.articles) % 25 == 0:
                self._save_progress()
                elapsed = time.time() - self.start_time
                articles_per_sec = (len(self.articles) - self.initial_count) / elapsed if elapsed > 0 else 0
                rem_articles = self.target - len(self.articles)
                eta_sec = rem_articles / articles_per_sec if articles_per_sec > 0 else 0
                
                logger.info(f"Progress: {len(self.articles)}/{self.target} | Errors: {self.errors} | Dupes: {self.dupes} | Speed: {articles_per_sec:.2f} it/s | ETA: {int(eta_sec//60)}m {int(eta_sec%60)}s")

    def run(self):
        self._load_existing_data()
        if len(self.articles) >= self.target:
            logger.info("Target already reached!")
            return

        logger.info("Phase 2: Extracting URLs from homepages using newspaper3k...")
        all_candidates = []
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._extract_urls_from_domain, domain): domain for domain in self.DOMAINS}
            for future in as_completed(futures):
                domain = futures[future]
                urls = future.result()
                new_urls = [u for u in urls if u not in self.seen_urls]
                all_candidates.extend(new_urls)
                logger.info(f"  {domain}: Found {len(urls)} URLs ({len(new_urls)} new)")
                
        logger.info(f"Total new candidate URLs: {len(all_candidates)}")
        random.shuffle(all_candidates)
        
        logger.info(f"Phase 2: Downloading articles in PARALLEL (target: {self.target})...")
        self.start_time = time.time()
        self.initial_count = len(self.articles)
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for c in all_candidates:
                if len(self.articles) >= self.target:
                    break
                futures.append(executor.submit(self._process_article, c))
                
            for future in as_completed(futures):
                if len(self.articles) >= self.target:
                    break

        self._save_progress()
        logger.info(f"DONE: {len(self.articles)} articles saved to {self.output_file}")

if __name__ == "__main__":
    scraper = ParallelNewsScraper(target=15000, output_file="data/news_10k_clean.json", workers=20)
    scraper.run()
