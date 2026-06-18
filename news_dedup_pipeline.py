import json
import os
import sys
import argparse
import random
import re
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from abc import ABC, abstractmethod

import faiss
import numpy as np
import networkx as nx
from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import adjusted_rand_score
import torch

# Fix for printing Russian characters in Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TQDM_NCOLS"] = "100"  # Fix progress bar spam in Windows console

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
    def __init__(self, model_name_or_path: str = "models/rubert-tiny2"):
        self.model_path = model_name_or_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_path, device=device)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        print(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
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
                
        if self.dup_data_path and os.path.exists(self.dup_data_path):
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
            if 'true_cluster_id' not in item or item['true_cluster_id'] is None:
                item['true_cluster_id'] = item.get('id')
            try:
                articles.append(NewsArticle(**item))
            except ValidationError:
                pass # ignore invalid
                
        print(f"Loaded and validated {len(articles)} articles out of {len(combined)}.")
        return articles

class MinHashPreFilter:
    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\w+', text.lower())
        return [" ".join(words[i:i+3]) for i in range(max(1, len(words) - 2))]

    def deduplicate(self, articles: List[NewsArticle]) -> Tuple[List[NewsArticle], Dict[str, List[NewsArticle]], Dict[str, MinHash]]:
        print(f"Running MinHash Pre-filtering on {len(articles)} articles...")
        
        groups = {}
        minhashes = {}
        for a in articles:
            m = MinHash(num_perm=self.num_perm)
            for word in self._tokenize(f"{a.title} {a.text}"):
                m.update(word.encode('utf8'))
            sig = tuple(m.hashvalues)
            if sig not in groups:
                groups[sig] = []
            groups[sig].append(a)
            minhashes[str(a.id)] = m
            
        unique_articles = []
        hidden_map = {}
        
        for sig, group in groups.items():
            group.sort(key=lambda x: x.published_at)
            rep = group[0]
            unique_articles.append(rep)
            if len(group) > 1:
                hidden_map[str(rep.id)] = group[1:]
                
        print(f"MinHash Pre-filter reduced dataset from {len(articles)} to {len(unique_articles)} articles.")
        return unique_articles, hidden_map, minhashes

class WindowedSimilaritySearch:
    def __init__(self, threshold: float = 0.98, time_window_days: int = 1, minhash_threshold: float = 0.10):
        self.threshold = threshold
        self.time_window_days = time_window_days
        self.minhash_threshold = minhash_threshold
        
    def search_pairs(self, embeddings: np.ndarray, articles: List[NewsArticle], minhashes: Dict[str, MinHash] = None) -> List[Tuple[int, int]]:
        print("Searching for pairs using Sliding Window...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / np.where(norms == 0, 1e-10, norms)
        emb_t = torch.tensor(embeddings_normalized, device=device)
        
        timestamps = [a.published_at for a in articles]
        pair_indices = []
        n = len(articles)
        
        from tqdm import tqdm
        for i in tqdm(range(n), desc="Sliding Window Search"):
            j_end = i + 1
            while j_end < n and (timestamps[j_end] - timestamps[i]).days <= self.time_window_days:
                j_end += 1
                
            if j_end > i + 1:
                window_embs = emb_t[i+1:j_end]
                sims = torch.matmul(window_embs, emb_t[i])
                valid_local_indices = torch.nonzero(sims >= self.threshold).squeeze(1).cpu().tolist()
                
                if isinstance(valid_local_indices, int):
                    valid_local_indices = [valid_local_indices]
                    
                for local_idx in valid_local_indices:
                    idx_j = i + 1 + local_idx
                    # Apply MinHash Veto
                    if minhashes is not None:
                        m_i = minhashes.get(str(articles[i].id))
                        m_j = minhashes.get(str(articles[idx_j].id))
                        if m_i and m_j:
                            if m_i.jaccard(m_j) < self.minhash_threshold:
                                continue # Veto this pair
                    pair_indices.append((i, idx_j))
                    
        print(f"Found {len(pair_indices)} candidate pairs.")
        return pair_indices

class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str = "BAAI/bge-reranker-v2-m3", threshold: float = 0.5):
        self.threshold = threshold
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name_or_path, max_length=512, device=device)

    def filter_pairs(self, articles: List[NewsArticle], pair_indices: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        print("Reranking pairs with Cross-Encoder...")
        
        if not pair_indices:
            return []
            
        pairs = []
        for (i, j) in pair_indices:
            text_i = f"{articles[i].title}. {articles[i].text}"
            text_j = f"{articles[j].title}. {articles[j].text}"
            pairs.append([text_i, text_j])
            
        print(f"Scoring {len(pairs)} candidate pairs...")
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=True)
        
        approved_pairs = []
        for idx, (u, v) in enumerate(pair_indices):
            score = 1 / (1 + np.exp(-scores[idx])) 
            if score >= self.threshold:
                approved_pairs.append((u, v, float(score)))
                
        return approved_pairs


class GraphClusterer:
    def build_clusters(self, articles: List[NewsArticle], approved_pairs: List[Tuple[int, int, float]]) -> List[EventCluster]:
        G = nx.Graph()
        G.add_nodes_from(range(len(articles)))
        
        for edge in approved_pairs:
            if len(edge) == 3:
                u, v, weight = edge
            else:
                u, v = edge[:2]
                weight = 1.0
            G.add_edge(u, v, weight=weight)
            
        if G.number_of_edges() > 0:
            # Используем алгоритм Лувена для предотвращения "снежного кома".
            # resolution > 1 разбивает граф на более мелкие/плотные кластеры.
            communities = nx.community.louvain_communities(G, weight='weight', resolution=1.0)
        else:
            communities = [{n} for n in G.nodes()]
            
        print(f"Found {len(communities)} clusters.")
        
        clusters = []
        for cluster_id, comp in enumerate(communities):
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
        search_engine: WindowedSimilaritySearch,
        clusterer: GraphClusterer,
        evaluator: MetricEvaluator,
        reranker: Optional[CrossEncoderReranker] = None,
        prefilter: Optional[MinHashPreFilter] = None
    ):
        self.embedder = embedder
        self.search_engine = search_engine
        self.clusterer = clusterer
        self.evaluator = evaluator
        self.reranker = reranker
        self.prefilter = prefilter
        
    def process(self, original_articles: List[NewsArticle]) -> List[EventCluster]:
        if not original_articles:
            return []
            
        articles = list(original_articles)
        hidden_map = {}
        minhashes = None
        if self.prefilter:
            articles, hidden_map, minhashes = self.prefilter.deduplicate(articles)
            
        print("Sorting articles by published_at for sliding window...")
        articles.sort(key=lambda x: x.published_at)
            
        print("Generating embeddings...")
        cache_path = "data/embeddings_cache.npy"
        texts = [f"{a.title}. {a.text}" for a in articles]
        
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}...")
            embeddings = np.load(cache_path)
            if len(embeddings) != len(articles):
                print("Cache size mismatch, recomputing...")
                embeddings = self.embedder.encode(texts)
                np.save(cache_path, embeddings)
        else:
            embeddings = self.embedder.encode(texts)
            np.save(cache_path, embeddings)
        
        print("Поиск кандидатов: Sliding Window Similarity Search.")
        pair_indices = self.search_engine.search_pairs(embeddings, articles, minhashes=minhashes)
        
        if self.reranker:
            approved_pairs = self.reranker.filter_pairs(articles, pair_indices)
        else:
            approved_pairs = [(u, v, 1.0) for u, v in pair_indices]
            
        clusters = self.clusterer.build_clusters(articles, approved_pairs)
        
        if hidden_map:
            print(f"Restoring hidden exact duplicates to clusters...")
            for cluster in clusters:
                restored = []
                for a in cluster.articles:
                    if str(a.id) in hidden_map:
                        restored.extend(hidden_map[str(a.id)])
                cluster.articles.extend(restored)
                
        self.evaluator.evaluate(original_articles, clusters)
        return clusters

def main():
    parser = argparse.ArgumentParser(description="News Deduplication Pipeline")
    parser.add_argument("--input", type=str, default="data/news_dump_full.json", help="Input JSON file with articles")
    parser.add_argument("--model_path", type=str, default="models/multilingual-e5-base", help="Path to the embedding model")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of news articles to process")
    parser.add_argument("--threshold", type=float, default=0.98, help="Cosine similarity threshold for FAISS clustering")
    parser.add_argument("--reranker_path", type=str, default="BAAI/bge-reranker-v2-m3", help="Path or name of the cross-encoder")
    parser.add_argument("--cross_threshold", type=float, default=0.5, help="Threshold for the cross-encoder")

    args = parser.parse_args()

    # Инициализация компонентов
    data_loader = DataLoader(args.input, None)
    embedder = RubertEmbeddingModel(model_name_or_path=args.model_path)
    search_engine = WindowedSimilaritySearch(threshold=args.threshold, time_window_days=1)
    clusterer = GraphClusterer()
    evaluator = MetricEvaluator()
    reranker = CrossEncoderReranker(model_name_or_path=args.reranker_path, threshold=args.cross_threshold) if args.reranker_path else None
    prefilter = MinHashPreFilter()
    
    pipeline = NewsDeduplicationPipeline(embedder, search_engine, clusterer, evaluator, reranker, prefilter)
    
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
    
    # Save cleaned dataset (only canonical articles)
    cleaned_articles = [c.canonical_article.to_dict() for c in clusters]
    with open("data/news_dump_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
    print(f"✅ Сохранен очищенный датасет без мусора ({len(cleaned_articles)} статей) в data/news_dump_cleaned.json")
    
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
