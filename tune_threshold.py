import numpy as np
from news_dedup_pipeline import DataLoader, RubertEmbeddingModel, FaissSearchEngine, GraphClusterer, MetricEvaluator

def tune():
    data_loader = DataLoader()
    articles = data_loader.load_data(limit=1000)
    
    embedder = RubertEmbeddingModel()
    texts = [f"{a.title}. {a.text}" for a in articles]
    embeddings = embedder.encode(texts)
    
    best_ari = -1.0
    best_threshold = 0.0
    
    print("Starting threshold tuning...")
    for th in np.arange(0.85, 1.00, 0.01):
        search_engine = FaissSearchEngine(threshold=th)
        lims, D, I = search_engine.search_pairs(embeddings)
        
        clusterer = GraphClusterer(time_window_days=3)
        clusters = clusterer.build_clusters(articles, lims, I)
        
        # Calculate ARI silently
        pred_labels = np.zeros(len(articles))
        id_to_idx = {a.id: i for i, a in enumerate(articles)}
        for cluster in clusters:
            for article in cluster.articles:
                pred_labels[id_to_idx[article.id]] = cluster.cluster_id
                
        true_labels = [a.true_cluster_id if a.true_cluster_id is not None else -1 for a in articles]
        valid_idx = [i for i, t in enumerate(true_labels) if t != -1]
        
        if valid_idx:
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score([true_labels[i] for i in valid_idx], [pred_labels[i] for i in valid_idx])
            print(f"Threshold: {th:.2f} -> Clusters: {len(clusters)} | ARI: {ari:.4f}")
            if ari > best_ari:
                best_ari = ari
                best_threshold = th
                
    print(f"\n✅ Best Threshold: {best_threshold:.2f} with ARI: {best_ari:.4f}")

if __name__ == '__main__':
    tune()
