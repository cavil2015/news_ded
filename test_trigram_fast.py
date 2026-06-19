import re
from news_dedup_pipeline import DataLoader, CachedSentenceTransformerModel, WindowedSimilaritySearch, GraphClusterer, MetricEvaluator, MinHash

def test():
    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    original_articles.sort(key=lambda x: x.published_at)
    
    # Compute trigram MinHashes WITHOUT deduplicating the list (to keep cache valid)
    minhashes = {}
    for a in original_articles:
        m = MinHash(num_perm=128)
        words = re.findall(r'\w+', a.get_full_text().lower())
        if len(words) < 3:
            trigrams = words
        else:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            
        for trigram in trigrams:
            m.update(trigram.encode('utf8'))
        minhashes[str(a.id)] = m
        
    texts = [a.get_full_text() for a in original_articles]
    embedder = CachedSentenceTransformerModel()
    embeddings = embedder.encode(texts)
    
    for thresh in [0.96, 0.97, 0.98]:
        for mh_thresh in [0.05, 0.1, 0.2]:
            print(f"\n--- Testing E5={thresh}, Trigram Jaccard={mh_thresh} ---")
            searcher = WindowedSimilaritySearch(threshold=thresh, time_window_days=2, minhash_threshold=mh_thresh)
            pairs = searcher.search_pairs(embeddings, original_articles, minhashes=minhashes)
            
            clusterer = GraphClusterer()
            clusters = clusterer.build_clusters(original_articles, [(u,v,1.0) for u,v in pairs])
            
            evaluator = MetricEvaluator()
            evaluator.evaluate(original_articles, clusters)

if __name__ == '__main__':
    test()
