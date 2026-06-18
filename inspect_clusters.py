import json
import random
import argparse
import sys

# Fix for printing Russian characters in Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

def inspect_clusters(full_data_path: str, clusters_path: str, seed: int = 42):
    try:
        with open(full_data_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"Error loading {full_data_path}: {e}")
        return
        
    article_dict = {str(item["id"]): item for item in full_data}
    
    try:
        with open(clusters_path, "r", encoding="utf-8") as f:
            clusters = json.load(f)
    except Exception as e:
        print(f"Error loading {clusters_path}: {e}")
        return
        
    # Find clusters with more than 1 article
    dup_clusters = [c for c in clusters if len(c["news_ids"]) > 1]
    
    print(f"Total clusters with duplicates: {len(dup_clusters)}")
    
    if not dup_clusters:
        print("No duplicates found.")
        return
        
    # Pick a few random clusters to show
    random.seed(seed)
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
    parser = argparse.ArgumentParser(description="Inspect news deduplication clusters")
    parser.add_argument("--full_data", type=str, default="data/news_dump_full.json", help="Path to full raw dataset")
    parser.add_argument("--clusters", type=str, default="data/clusters_output.json", help="Path to clusters output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    
    inspect_clusters(args.full_data, args.clusters, args.seed)
