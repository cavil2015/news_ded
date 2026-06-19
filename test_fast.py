from news_dedup_pipeline import DataLoader, CachedSentenceTransformerModel, WindowedSimilaritySearch, GraphClusterer, MetricEvaluator, MinHashPreFilter
import time

def test():
    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    
    prefilter = MinHashPreFilter()
    articles, hidden, minhashes = prefilter.deduplicate(original_articles)
    articles.sort(key=lambda x: x.published_at)
    
    texts = [a.get_full_text() for a in articles]
    embedder = CachedSentenceTransformerModel()
    embeddings = embedder.encode(texts)
    
    for thresh in [0.97, 0.975, 0.98]:
        for mh_thresh in [0.7, 0.75, 0.8, 0.85]:
            print(f"\n--- Testing E5={thresh}, MinHash Jaccard={mh_thresh} ---")
            searcher = WindowedSimilaritySearch(threshold=thresh, time_window_days=2, minhash_threshold=mh_thresh)
            pairs = searcher.search_pairs(embeddings, articles, minhashes=minhashes)
            
            clusterer = GraphClusterer()
            clusters = clusterer.build_clusters(articles, [(u,v,1.0) for u,v in pairs])
            
            if hidden:
                for cluster in clusters:
                    restored = []
                    for a in cluster.articles:
                        if str(a.id) in hidden:
                            restored.extend(hidden[str(a.id)])
                    cluster.articles.extend(restored)
            
            evaluator = MetricEvaluator()
            evaluator.evaluate(original_articles, clusters)

if __name__ == '__main__':
    test()
