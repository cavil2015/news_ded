import re
from news_dedup_pipeline import DataLoader, CachedSentenceTransformerModel, WindowedSimilaritySearch, GraphClusterer, MetricEvaluator, MinHashPreFilter, MinHash

def test():
    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    
    # Monkey-patch MinHashPreFilter to use character 5-grams or word 3-grams
    def _tokenize(self, text: str):
        words = re.findall(r'\w+', text.lower())
        # Word trigrams
        if len(words) < 3:
            return words
        return [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
    MinHashPreFilter._tokenize = _tokenize
    
    prefilter = MinHashPreFilter()
    articles, hidden, minhashes = prefilter.deduplicate(original_articles)
    articles.sort(key=lambda x: x.published_at)
    
    texts = [a.get_full_text() for a in articles]
    embedder = CachedSentenceTransformerModel()
    embeddings = embedder.encode(texts)
    
    for thresh in [0.96, 0.97, 0.98]:
        for mh_thresh in [0.1, 0.2, 0.3, 0.4]: # Jaccard will be lower for trigrams
            print(f"\n--- Testing E5={thresh}, Trigram Jaccard={mh_thresh} ---")
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
