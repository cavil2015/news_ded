from news_dedup_pipeline import DataLoader, CachedSentenceTransformerModel, WindowedSimilaritySearch, GraphClusterer, MetricEvaluator, MinHashPreFilter, CrossEncoderReranker, NewsDeduplicationPipeline
import time
import torch

def test():
    # Monkey-patch CrossEncoderReranker
    original_filter = CrossEncoderReranker.filter_pairs
    
    def modified_filter(self, articles, candidate_pairs):
        print(f"Reranking {len(candidate_pairs)} candidate pairs with [:2000] truncation...")
        text_pairs = []
        for u, v in candidate_pairs:
            text_u = articles[u].get_full_text()[:2000]
            text_v = articles[v].get_full_text()[:2000]
            text_pairs.append((text_u, text_v))
            
        approved_pairs = []
        batch_size = 32
        
        scores = self.model.predict(text_pairs, batch_size=batch_size, show_progress_bar=True)
        
        for i, score in enumerate(scores):
            # Sigmoid conversion
            import math
            prob = 1 / (1 + math.exp(-score))
            if prob >= self.threshold:
                u, v = candidate_pairs[i]
                approved_pairs.append((u, v, float(prob)))
                    
        print(f"Approved {len(approved_pairs)} pairs out of {len(candidate_pairs)}")
        return approved_pairs
        
    CrossEncoderReranker.filter_pairs = modified_filter

    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    
    embedder = CachedSentenceTransformerModel()
    # Use E5=0.985 to reduce candidate pairs to ~6500, which will take ~1 minute to rerank
    search_engine = WindowedSimilaritySearch(threshold=0.985, time_window_days=2, minhash_threshold=0.0)
    clusterer = GraphClusterer()
    evaluator = MetricEvaluator()
    reranker = CrossEncoderReranker(model_name_or_path="models/bge-reranker-v2-m3", threshold=0.8)
    prefilter = MinHashPreFilter()
    
    pipeline = NewsDeduplicationPipeline(embedder, search_engine, clusterer, evaluator, reranker, prefilter)
    
    start = time.time()
    clusters = pipeline.process(original_articles)
    end = time.time()
    
    print(f"Total time: {end - start:.2f} seconds")

if __name__ == '__main__':
    test()
