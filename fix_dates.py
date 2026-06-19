import json
from datetime import datetime, timezone
import dateutil.parser

def fix():
    with open("data/news_15k_ready.json", "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    for a in articles:
        if a.get("published_at"):
            try:
                # Use dateutil to handle various formats like "Tue, 14 Nov 2023 10:00:00 +0000"
                d = dateutil.parser.parse(a["published_at"])
                if d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
                a["published_at"] = d.isoformat()
            except Exception:
                pass
                
    with open("data/news_15k_ready.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
        
    print("Fixed dates.")

if __name__ == "__main__":
    fix()
