import json
import os
import sys
import argparse
import random
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from abc import ABC, abstractmethod

import faiss
import numpy as np
import networkx as nx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score

# Fix for printing Russian characters in Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Data Models ---
class NewsArticle(BaseModel):
    id: int | str
    title: str
    text: str
    source: str
    published_at: datetime
    true_cluster_id: Optional[int] = Field(default=None)

class EventCluster:
    def __init__(self, cluster_id: int, articles: List[NewsArticle]):
        self.cluster_id = cluster_id
        self.articles = articles
        
    @property
    def canonical_article(self) -> NewsArticle:
        """Selects the earliest article as the canonical source."""
        return min(self.articles, key=lambda a: a.published_at)
        
    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "canonical_news_id": self.canonical_article.id,
            "news_ids": [a.id for a in self.articles]
        }

# --- Interfaces ---
class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass

# --- Implementations ---
class RubertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name_or_path: str = "cointegrated/rubert-tiny2"):
        local_path = os.path.join(os.path.dirname(__file__), "models", "rubert-tiny2")
        self.model_path = local_path if os.path.exists(local_path) else model_name_or_path
        self.model = SentenceTransformer(self.model_path)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.ascontiguousarray(embeddings, dtype=np.float32)

class DataLoader:
    def __init__(self, full_data_path: str = "data/news_dump_full.json", dup_data_path: str = "data/synthetic_duplicates.json"):
        self.full_data_path = full_data_path
        self.dup_data_path = dup_data_path

    def load_data(self, limit: Optional[int] = None) -> List[NewsArticle]:
        data_full, data_dup = [], []
        if os.path.exists(self.full_data_path):
            with open(self.full_data_path, "r", encoding="utf-8") as f:
                data_full = json.load(f)
                
        if os.path.exists(self.dup_data_path):
            with open(self.dup_data_path, "r", encoding="utf-8") as f:
                data_dup = json.load(f)
                
        if limit is not None:
            half = limit // 2
            data_full = data_full[:half]
            data_dup = data_dup[:half]
            
        combined = data_full + data_dup
        random.shuffle(combined)
        
        articles = []
        for item in combined:
            try:
                articles.append(NewsArticle(**item))
            except ValidationError:
                pass # ignore invalid
                
        print(f"Loaded and validated {len(articles)} articles out of {len(combined)}.")
        return articles

class FaissSearchEngine:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        
    def search_pairs(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        lims, D, I = index.range_search(embeddings, self.threshold)
        return lims, D, I

class GraphClusterer:
    def __init__(self, time_window_days: int = 3):
        self.time_window_days = time_window_days
        
    def build_clusters(self, articles: List[NewsArticle], lims: np.ndarray, I: np.ndarray) -> List[EventCluster]:
        G = nx.Graph()
        G.add_nodes_from(range(len(articles)))
        
        for i in range(len(articles)):
            start = lims[i]
            end = lims[i+1]
            for j in range(start, end):
                if i != I[j]:
                    # Time Window Heuristic
                    time_diff = abs((articles[i].published_at - articles[I[j]].published_at).days)
                    if time_diff <= self.time_window_days:
                        G.add_edge(i, I[j])
                    
        components = list(nx.connected_components(G))
        print(f"Found {len(components)} clusters.")
        
        clusters = []
        for cluster_id, comp in enumerate(components):
            cluster_articles = [articles[node] for node in comp]
            clusters.append(EventCluster(cluster_id=cluster_id, articles=cluster_articles))
            
        return clusters

class MetricEvaluator:
    @staticmethod
    def evaluate(articles: List[NewsArticle], clusters: List[EventCluster]) -> Optional[float]:
        pred_labels = np.zeros(len(articles))
        # Create mapping from article id to index
        id_to_idx = {a.id: i for i, a in enumerate(articles)}
        
        for cluster in clusters:
            for article in cluster.articles:
                pred_labels[id_to_idx[article.id]] = cluster.cluster_id
                
        true_labels = [a.true_cluster_id if a.true_cluster_id is not None else -1 for a in articles]
        valid_idx = [i for i, t in enumerate(true_labels) if t != -1]
        
        if valid_idx:
            ari = adjusted_rand_score([true_labels[i] for i in valid_idx], [pred_labels[i] for i in valid_idx])
            print(f"✅ Качество алгоритма (Adjusted Rand Index): {ari:.4f}")
            return ari
        else:
            print("No valid true_cluster_id found for metric evaluation.")
            return None

# --- Orchestrator ---
class NewsDeduplicationPipeline:
    def __init__(
        self,
        embedder: BaseEmbeddingModel,
        search_engine: FaissSearchEngine,
        clusterer: GraphClusterer,
        evaluator: MetricEvaluator
    ):
        self.embedder = embedder
        self.search_engine = search_engine
        self.clusterer = clusterer
        self.evaluator = evaluator
        
    def process(self, articles: List[NewsArticle]) -> List[EventCluster]:
        if not articles:
            return []
            
        print("Generating embeddings...")
        texts = [f"{a.title}. {a.text}" for a in articles]
        embeddings = self.embedder.encode(texts)
        
        print("FAISS Кластеризация: Построение графа Connected Components и поиск соседей.")
        lims, D, I = self.search_engine.search_pairs(embeddings)
        clusters = self.clusterer.build_clusters(articles, lims, I)
        
        self.evaluator.evaluate(articles, clusters)
        return clusters

def main():
    parser = argparse.ArgumentParser(description="News Deduplication Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of news articles to process")
    args = parser.parse_args()

    # Dependency Injection
    data_loader = DataLoader()
    embedder = RubertEmbeddingModel()
    search_engine = FaissSearchEngine(threshold=0.99)
    clusterer = GraphClusterer(time_window_days=3)
    evaluator = MetricEvaluator()
    
    pipeline = NewsDeduplicationPipeline(embedder, search_engine, clusterer, evaluator)
    
    articles = data_loader.load_data(limit=args.limit)
    clusters = pipeline.process(articles)
    
    if not clusters:
        print("No clusters generated.")
        return
        
    # Save Output
    os.makedirs("data", exist_ok=True)
    output_data = [c.to_dict() for c in clusters]
    with open("data/clusters_output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print("\n✅ Сохранено разбиение по кластерам в data/clusters_output.json")
    
    # Print top clusters
    print("\n✅ Вывод топовых кластеров:")
    clusters_sorted = sorted(clusters, key=lambda c: len(c.articles), reverse=True)[:5]
    for i, cluster in enumerate(clusters_sorted):
        canonical = cluster.canonical_article
        print(f"\nCluster {i+1} (Size: {len(cluster.articles)}):")
        print(f"🔹 Каноничный Заголовок: {canonical.title}")
        print(f"🔹 Каноничный ID: {canonical.id}")
        
    print("\nPipeline successfully completed.")

if __name__ == "__main__":
    main()
