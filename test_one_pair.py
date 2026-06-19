from news_dedup_pipeline import DataLoader
from sentence_transformers import CrossEncoder

def test():
    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    
    # Find a true duplicate pair
    true_clusters = {}
    for a in original_articles:
        if a.true_cluster_id not in true_clusters:
            true_clusters[a.true_cluster_id] = []
        true_clusters[a.true_cluster_id].append(a)
        
    pairs = [v for k,v in true_clusters.items() if len(v) > 1]
    
    model = CrossEncoder("models/bge-reranker-v2-m3", max_length=512)
    
    print("Testing true pairs:")
    for i in range(5):
        a, b = pairs[i]
        text_a = a.get_full_text()[:2000]
        text_b = b.get_full_text()[:2000]
        score = model.predict([[text_a, text_b]])[0]
        print(f"Pair {i}: ID {a.id} vs {b.id} | Score: {score}")

if __name__ == '__main__':
    test()
