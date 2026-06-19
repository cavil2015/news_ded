from news_dedup_pipeline import DataLoader, CachedSentenceTransformerModel, WindowedSimilaritySearch
from sentence_transformers import CrossEncoder

def test():
    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    original_articles.sort(key=lambda x: x.published_at)
    
    embedder = CachedSentenceTransformerModel()
    texts = [a.get_full_text() for a in original_articles]
    embeddings = embedder.encode(texts)
    
    searcher = WindowedSimilaritySearch(threshold=0.99, time_window_days=2, minhash_threshold=0.0)
    pairs = searcher.search_pairs(embeddings, original_articles)
    
    # Filter false positives
    fp_pairs = []
    for u, v in pairs:
        a, b = original_articles[u], original_articles[v]
        if a.true_cluster_id != b.true_cluster_id:
            fp_pairs.append((a, b))
            
    print(f"Found {len(fp_pairs)} false positives.")
    
    model = CrossEncoder("models/bge-reranker-v2-m3", max_length=512)
    
    print("Testing FP pairs:")
    for i in range(5):
        a, b = fp_pairs[i]
        text_a = a.get_full_text()[:2000]
        text_b = b.get_full_text()[:2000]
        score = model.predict([[text_a, text_b]])[0]
        print(f"FP Pair {i}: Score: {score:.4f}")

if __name__ == '__main__':
    test()
