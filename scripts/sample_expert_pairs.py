import json
import random
import os
from itertools import combinations

def load_data():
    with open('data/news_dump_polluted.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    article_dict = {a['id']: a for a in articles}
    
    with open('data/clusters_polluted.json', 'r', encoding='utf-8') as f:
        clusters = json.load(f)
        
    return article_dict, clusters

def main():
    print("Loading data...")
    article_dict, clusters = load_data()
    
    # Extract positive pairs (same cluster)
    positive_pairs = []
    for c in clusters:
        # Filter out synthetic duplicates (id >= 1000000) and ensure they exist
        natural_ids = [nid for nid in c['news_ids'] if nid < 1000000 and nid in article_dict]
        if len(natural_ids) > 1:
            # Add all possible pairs within the cluster
            positive_pairs.extend(list(combinations(natural_ids, 2)))
            
    print(f"Total possible natural positive pairs: {len(positive_pairs)}")
    
    # Sample 500 positives
    if len(positive_pairs) > 500:
        positives = random.sample(positive_pairs, 500)
    else:
        positives = positive_pairs
        
    print(f"Selected {len(positives)} positive pairs.")
    
    # Extract negative pairs (different clusters)
    negative_pairs = []
    all_natural_ids = [aid for aid in article_dict.keys() if aid < 1000000]
    
    # Create a fast lookup for cluster ID by news ID to ensure they are in different clusters
    id_to_cluster = {}
    for c in clusters:
        for nid in c['news_ids']:
            id_to_cluster[nid] = c['cluster_id']
            
    print("Sampling negative pairs...")
    while len(negative_pairs) < 500:
        a = random.choice(all_natural_ids)
        b = random.choice(all_natural_ids)
        if a != b and id_to_cluster.get(a) != id_to_cluster.get(b):
            negative_pairs.append((a, b))
            
    print(f"Selected {len(negative_pairs)} negative pairs.")
    
    # Build final dataset
    final_dataset = []
    
    # We will pass the title + content to the LLM
    def build_text(aid):
        a = article_dict[aid]
        return f"TITLE: {a.get('title', '')}\n\nCONTENT:\n{a.get('content', '')}"
        
    pair_id = 0
    for a, b in positives:
        final_dataset.append({
            "pair_id": pair_id,
            "news_id_a": a,
            "news_id_b": b,
            "text_a": build_text(a),
            "text_b": build_text(b),
            "algorithmic_label": 1  # Pipeline says DUPLICATE
        })
        pair_id += 1
        
    for a, b in negative_pairs:
        final_dataset.append({
            "pair_id": pair_id,
            "news_id_a": a,
            "news_id_b": b,
            "text_a": build_text(a),
            "text_b": build_text(b),
            "algorithmic_label": 0  # Pipeline says NOT DUPLICATE
        })
        pair_id += 1
        
    # Shuffle the dataset so LLM doesn't see all positives then all negatives
    random.shuffle(final_dataset)
    
    output_path = 'data/llm_markup_dataset.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(final_dataset)} pairs to {output_path}")

if __name__ == '__main__':
    main()
