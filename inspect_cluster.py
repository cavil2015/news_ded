import json
import argparse
import sys

# Fix for printing Russian characters in Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

def inspect_largest_cluster(clusters_path: str, full_data_path: str, synthetic_data_path: str, limit: int = 20):
    """
    Finds the largest cluster in the clustering output and prints its articles.
    """
    try:
        with open(clusters_path, "r", encoding="utf-8") as f:
            clusters = json.load(f)
    except Exception as e:
        print(f"Error loading {clusters_path}: {e}")
        return

    if not clusters:
        print("No clusters found.")
        return

    largest = max(clusters, key=lambda c: len(c.get('news_ids', [])))
    print(f"Largest Cluster Size: {len(largest.get('news_ids', []))}\n")

    data = []
    
    # Load original data
    try:
        with open(full_data_path, "r", encoding="utf-8") as f:
            data.extend(json.load(f))
    except Exception as e:
        print(f"Warning: Could not load {full_data_path}: {e}")
        
    # Load synthetic data
    if synthetic_data_path:
        try:
            with open(synthetic_data_path, "r", encoding="utf-8") as f:
                data.extend(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load {synthetic_data_path}: {e}")

    ndict = {str(i['id']): i.get('title', 'No Title') for i in data}

    for nid in largest.get('news_ids', [])[:limit]:
        print(f"[{nid}] - {ndict.get(str(nid), 'Unknown Article')}")
        
    if len(largest.get('news_ids', [])) > limit:
        print(f"... and {len(largest['news_ids']) - limit} more.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the largest cluster")
    parser.add_argument("--clusters", type=str, default="data/clusters_output.json", help="Path to clusters output")
    parser.add_argument("--full_data", type=str, default="data/news_dump_full.json", help="Path to original data")
    parser.add_argument("--synthetic_data", type=str, default="data/synthetic_duplicates.json", help="Path to synthetic data")
    parser.add_argument("--limit", type=int, default=20, help="Max titles to show")
    
    args = parser.parse_args()
    inspect_largest_cluster(args.clusters, args.full_data, args.synthetic_data, args.limit)
