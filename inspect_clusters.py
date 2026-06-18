import json
import random

def inspect_clusters():
    with open("data/news_dump_full.json", "r", encoding="utf-8") as f:
        full_data = json.load(f)
    article_dict = {str(item["id"]): item for item in full_data}
    
    with open("data/clusters_output.json", "r", encoding="utf-8") as f:
        clusters = json.load(f)
        
    # Find clusters with more than 1 article
    dup_clusters = [c for c in clusters if len(c["news_ids"]) > 1]
    
    print(f"Total clusters with duplicates: {len(dup_clusters)}")
    
    if not dup_clusters:
        print("No duplicates found.")
        return
        
    # Pick a few random clusters to show
    random.seed(42)  # For reproducibility
    sample_clusters = random.sample(dup_clusters, min(5, len(dup_clusters)))
    
    for i, c in enumerate(sample_clusters):
        print(f"\n{'='*50}")
        print(f"CLUSTER {i+1} (Size: {len(c['news_ids'])})")
        print(f"{'='*50}")
        
        for idx, news_id in enumerate(c["news_ids"]):
            str_id = str(news_id)
            if str_id in article_dict:
                article = article_dict[str_id]
                title = article.get("title", "")
                text = article.get("text", "")
                # truncate text to 200 chars
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"[{idx+1}] ID: {news_id} | Title: {title}")
                print(f"    Text snippet: {text}\n")
            else:
                print(f"[{idx+1}] ID: {news_id} (Not found in dict)")

if __name__ == "__main__":
    inspect_clusters()
