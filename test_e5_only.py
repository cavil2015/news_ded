from news_dedup_pipeline import DataLoader, CachedSentenceTransformerModel, WindowedSimilaritySearch, GraphClusterer, MetricEvaluator

def test():
    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    original_articles.sort(key=lambda x: x.published_at)
    
    texts = [a.get_full_text() for a in original_articles]
    embedder = CachedSentenceTransformerModel()
    embeddings = embedder.encode(texts)
    
    for thresh in [0.985, 0.99, 0.992, 0.995, 0.998, 0.999]:
        print(f"\n--- Testing E5={thresh} ---")
        searcher = WindowedSimilaritySearch(threshold=thresh, time_window_days=2, minhash_threshold=0.0)
        pairs = searcher.search_pairs(embeddings, original_articles)
        
        clusterer = GraphClusterer()
        clusters = clusterer.build_clusters(original_articles, [(u,v,1.0) for u,v in pairs])
        
        evaluator = MetricEvaluator()
        evaluator.evaluate(original_articles, clusters)

if __name__ == '__main__':
    test()
