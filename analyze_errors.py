from news_dedup_pipeline import DataLoader, CachedSentenceTransformerModel, WindowedSimilaritySearch, GraphClusterer, MetricEvaluator, MinHashPreFilter, CrossEncoderReranker, NewsDeduplicationPipeline

def main():
    loader = DataLoader('data/news_dump_full.json', 'data/synthetic_duplicates.json')
    original_articles = loader.load_data()
    
    # We run the pipeline quickly
    embedder = CachedSentenceTransformerModel()
    search_engine = WindowedSimilaritySearch(threshold=0.96, time_window_days=7, minhash_threshold=0.0)
    clusterer = GraphClusterer()
    evaluator = MetricEvaluator()
    reranker = CrossEncoderReranker(model_name_or_path="models/bge-reranker-v2-m3", threshold=0.6)
    prefilter = MinHashPreFilter()
    
    pipeline = NewsDeduplicationPipeline(embedder, search_engine, clusterer, evaluator, reranker, prefilter)
    clusters = pipeline.process(original_articles)
    
    ari, precision, recall, f1, fp_pairs, fn_pairs = evaluator.evaluate(original_articles, clusters)
    
    print("\n" + "="*50)
    print("TOP 5 FALSE POSITIVES (Algorithm thought they were duplicates, but they aren't)")
    print("="*50)
    id_to_article = {str(a.id): a for a in original_articles}
    for i, (u, v) in enumerate(fp_pairs[:5]):
        fp = (id_to_article[u], id_to_article[v])
        print(f"--- FP Pair {i+1} ---")
        print(f"Article A: {fp[0].title}. {fp[0].text[:800]}...\n")
        print(f"Article B: {fp[1].title}. {fp[1].text[:800]}...\n")
        
    print("\n" + "="*50)
    print("TOP 5 FALSE NEGATIVES (Algorithm missed these true duplicates)")
    print("="*50)
    for i, (u, v) in enumerate(fn_pairs[:5]):
        print(f"\n--- FN Pair {i+1} ---")
        print(f"Article A: {id_to_article[u].get_full_text()[:300]}...")
        print(f"Article B: {id_to_article[v].get_full_text()[:300]}...")

if __name__ == '__main__':
    main()
