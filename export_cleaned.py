import json
import os

def export_cleaned():
    print("Loading full dataset...")
    with open("data/news_dump_full.json", "r", encoding="utf-8") as f:
        full_data = json.load(f)
    
    # index by id
    article_dict = {str(item["id"]): item for item in full_data}
    
    print("Loading clusters...")
    with open("data/clusters_output.json", "r", encoding="utf-8") as f:
        clusters = json.load(f)
        
    cleaned_articles = []
    missing = 0
    for c in clusters:
        can_id = str(c["canonical_news_id"])
        if can_id in article_dict:
            cleaned_articles.append(article_dict[can_id])
        else:
            missing += 1
            
    print(f"Extracted {len(cleaned_articles)} canonical articles. Missing: {missing}")
    
    print("Saving clean dataset...")
    with open("data/news_dump_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    export_cleaned()
