from news_dedup_pipeline import DataLoader, CachedSentenceTransformerModel, WindowedSimilaritySearch, GraphClusterer, MetricEvaluator, MinHashPreFilter, CrossEncoderReranker, NewsDeduplicationPipeline
import time

def test():
    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    
    embedder = CachedSentenceTransformerModel()
    
    # E5 threshold 0.985 gives ~10,000 candidate pairs
    search_engine = WindowedSimilaritySearch(threshold=0.96, time_window_days=7, minhash_threshold=0.0)
    clusterer = GraphClusterer()
    evaluator = MetricEvaluator()
    
    # 3. Cross-Encoder reranking
    reranker = CrossEncoderReranker(model_name_or_path="models/bge-reranker-v2-m3", threshold=0.6)
    prefilter = MinHashPreFilter()
    
    pipeline = NewsDeduplicationPipeline(embedder, search_engine, clusterer, evaluator, reranker, prefilter)
    
    start = time.time()
    clusters = pipeline.process(original_articles)
    end = time.time()
    
    print(f"\n✅ Total Execution Time: {end - start:.2f} seconds")

if __name__ == '__main__':
    test()
