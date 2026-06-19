import json
from datetime import datetime

INPUT_FILE = "data/news_10k_clean.json"
OUTPUT_FILE = "data/news_15k_ready.json"

def prepare_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    now_str = datetime.utcnow().isoformat()
    
    for a in articles:
        if not a.get("published_at"):
            a["published_at"] = now_str
            
        if not a.get("title"):
            # Extract first sentence as title
            text = a.get("text", "")
            first_sentence = text.split(". ")[0][:100]
            if first_sentence:
                a["title"] = first_sentence + "..."
            else:
                a["title"] = "News Article"
                
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
        
    print(f"Prepared {len(articles)} articles and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_dataset()
