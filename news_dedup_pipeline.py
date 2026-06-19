import json
import os
import sys
import argparse
import random
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

import faiss
import numpy as np
import networkx as nx
from datasketch import MinHash
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
os.environ["TQDM_NCOLS"] = "100"

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Models ---
class NewsArticle(BaseModel):
    id: int | str
    title: str
    text: str
    source: str
    published_at: datetime
    true_cluster_id: Optional[int] = Field(default=None)

    def get_full_text(self) -> str:
        text = f"{self.title}. {self.text}"
        bad_phrases = [
            "Yahoo on osa", "Sivustot ja sovellukset", "Jos et halua meidän", 
            "Tietosuojakäytännöstämme", "Risk Disclosure", "Fusion Media", 
            "Access to this page has been denied", "Target URL returned error", 
            "Just a moment...", "Si no quieres que nosotros", "Tu privacidad", 
            "Para obtener más información", "Voit peruuttaa", "Prices of cryptocurrencies", 
            "evästeitä", "cookies", "tietosuoja", "privacidad", "hallintapaneelin",
            "Markdown Content: Before we continue", "Press & Hold to confirm you are a human",
            "Before deciding to trade in financial instrument", "Trading on margin increases the financial risks"
        ]
        
        sentences = text.split('. ')
        clean_sentences = []
        for s in sentences:
            if not any(bp.lower() in s.lower() for bp in bad_phrases):
                clean_sentences.append(s)
                
        # If everything was boilerplate, fallback to title
        clean_text = ". ".join(clean_sentences).strip()
        if not clean_text:
            return self.title
        return clean_text

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
class CachedSentenceTransformerModel(BaseEmbeddingModel):
    def __init__(self, model_name_or_path: str = "intfloat/multilingual-e5-base", cache_path: Optional[str] = "data/embeddings_cache.npy"):
        self.model_path = model_name_or_path
        self.cache_path = cache_path
        self.model = SentenceTransformer(self.model_path, device=get_device())
        
    def encode(self, texts: List[str]) -> np.ndarray:
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading cached embeddings from {self.cache_path}...")
            embeddings = np.load(self.cache_path)
            if len(embeddings) == len(texts):
                return embeddings
            print("Cache size mismatch, recomputing...")

        print(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, embeddings)
            
        return embeddings

class DataLoader:
    def __init__(self, full_data_path: str = "data/news_dump_full.json", dup_data_path: Optional[str] = "data/synthetic_duplicates.json"):
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
            half = limit // 2 if self.dup_data_path else limit
            data_full = data_full[:half]
            data_dup = data_dup[:half]
            
        combined = data_full + data_dup
        random.shuffle(combined)
        
        # Identifiers of articles that are part of the synthetic pairs
        ground_truth_ids = set()
        for item in data_dup:
            if 'true_cluster_id' in item:
                ground_truth_ids.add(str(item['true_cluster_id']))
            ground_truth_ids.add(str(item.get('id')))
            
        articles = []
        for item in combined:
            item_id = str(item.get('id'))
            # If the article is in the ground truth subset, assign a valid cluster ID
            if item_id in ground_truth_ids:
                if 'true_cluster_id' not in item or item['true_cluster_id'] is None:
                    item['true_cluster_id'] = item.get('id')
            else:
                # Mask out unannotated articles so they don't skew the evaluation
                item['true_cluster_id'] = None
                
            try:
                articles.append(NewsArticle(**item))
            except ValidationError:
                pass # ignore invalid
                
        print(f"Loaded and validated {len(articles)} articles out of {len(combined)}.")
        return articles

class MinHashPreFilter:
    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def deduplicate(self, articles: List[NewsArticle]) -> Tuple[List[NewsArticle], Dict[str, List[NewsArticle]], Dict[str, MinHash]]:
        print(f"Running MinHash Pre-filtering on {len(articles)} articles...")
        
        groups = {}
        minhashes = {}
        for a in articles:
            m = MinHash(num_perm=self.num_perm)
            for word in self._tokenize(a.get_full_text()):
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
    def __init__(self, threshold: float = 0.96, time_window_days: int = 2, minhash_threshold: float = 0.70):
        self.threshold = threshold
        self.time_window_days = time_window_days
        self.minhash_threshold = minhash_threshold
        
    def search_pairs(self, embeddings: np.ndarray, articles: List[NewsArticle], minhashes: Optional[Dict[str, MinHash]] = None) -> List[Tuple[int, int]]:
        print("Searching for pairs using Sliding Window...")
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / np.where(norms == 0, 1e-10, norms)
        emb_t = torch.tensor(embeddings_normalized, device=get_device())
        
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
    def __init__(self, model_name_or_path: str = "BAAI/bge-reranker-v2-m3", threshold: float = 0.4):
        self.threshold = threshold
        self.model = CrossEncoder(model_name_or_path, max_length=512, device=get_device())

    def filter_pairs(self, articles: List[NewsArticle], pair_indices: List[Tuple[int, int]], checkpoint_path: str = "data/cross_encoder_checkpoint.pkl") -> List[Tuple[int, int, float]]:
        import os, pickle
        if not pair_indices:
            return []
            
        approved_pairs = []
        start_idx = 0
        
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                total_pairs = checkpoint_data.get('total_pairs', -1)
                if total_pairs == len(pair_indices):
                    print(f"Loading Cross-Encoder progress from {checkpoint_path}")
                    approved_pairs = checkpoint_data.get('approved_pairs', [])
                    start_idx = checkpoint_data.get('processed_count', 0)
                else:
                    print("Candidate pairs count changed. Discarding old Cross-Encoder cache.")
                
        if start_idx >= len(pair_indices):
            print("All candidate pairs already reranked.")
            return approved_pairs
            
        print(f"Reranking {len(pair_indices)} candidate pairs (Resuming from {start_idx})...")
        
        chunk_size = 5000
        for i in range(start_idx, len(pair_indices), chunk_size):
            chunk_indices = pair_indices[i:i+chunk_size]
            pairs = [[articles[u].get_full_text()[:2000], articles[v].get_full_text()[:2000]] for u, v in chunk_indices]
            
            print(f"Processing chunk {i//chunk_size + 1}/{(len(pair_indices)+chunk_size-1)//chunk_size}...")
            scores = self.model.predict(pairs, batch_size=32, show_progress_bar=True)
            
            for idx, (u, v) in enumerate(chunk_indices):
                score = 1 / (1 + np.exp(-scores[idx])) 
                if score >= self.threshold:
                    approved_pairs.append((u, v, float(score)))
                    
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'approved_pairs': approved_pairs, 
                    'processed_count': i + len(chunk_indices),
                    'total_pairs': len(pair_indices)
                }, f)
                
        return approved_pairs


class GraphClusterer:
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution

    def build_clusters(self, articles: List[NewsArticle], approved_pairs: List[Tuple[int, int, float]]) -> List[EventCluster]:
        G = nx.Graph()
        G.add_nodes_from(range(len(articles)))
        
        for edge in approved_pairs:
            u, v = edge[:2]
            weight = edge[2] if len(edge) == 3 else 1.0
            G.add_edge(u, v, weight=weight)
            
        if G.number_of_edges() > 0:
            communities = nx.community.louvain_communities(G, weight='weight', resolution=self.resolution)
        else:
            communities = [{n} for n in G.nodes()]
            
        print(f"Found {len(communities)} clusters.")
        
        clusters = []
        for cluster_id, comp in enumerate(communities):
            cluster_articles = [articles[node] for node in comp]
            clusters.append(EventCluster(cluster_id=cluster_id, articles=cluster_articles))
            
        return clusters

class MetricEvaluator:
    def evaluate(self, original_articles: List[NewsArticle], clusters: List[EventCluster]):
        from sklearn.metrics import adjusted_rand_score
        
        true_labels = []
        pred_labels = []
        
        article_id_to_pred_cluster = {}
        for c in clusters:
            for a in c.articles:
                article_id_to_pred_cluster[str(a.id)] = c.cluster_id
                
        # To calculate Precision/Recall on pairs, we will count True Positives, False Positives, False Negatives
        true_pairs = set()
        pred_pairs = set()
        
        # Group by true cluster id to get true pairs
        true_clusters_dict = {}
        for a in original_articles:
            true_label = a.true_cluster_id if a.true_cluster_id is not None else -1
            if true_label == -1:
                continue
            if true_label not in true_clusters_dict:
                true_clusters_dict[true_label] = []
            true_clusters_dict[true_label].append(str(a.id))
            
            true_labels.append(true_label)
            pred_labels.append(article_id_to_pred_cluster.get(str(a.id), -1))
            
        for t_id, articles_in_cluster in true_clusters_dict.items():
            for i in range(len(articles_in_cluster)):
                for j in range(i+1, len(articles_in_cluster)):
                    u, v = sorted([articles_in_cluster[i], articles_in_cluster[j]])
                    true_pairs.add((u, v))
                    
        for c in clusters:
            articles_in_cluster = [str(a.id) for a in c.articles]
            for i in range(len(articles_in_cluster)):
                for j in range(i+1, len(articles_in_cluster)):
                    u, v = sorted([articles_in_cluster[i], articles_in_cluster[j]])
                    pred_pairs.add((u, v))
                    
        tp = len(true_pairs.intersection(pred_pairs))
        fp_pairs = pred_pairs - true_pairs
        fn_pairs = true_pairs - pred_pairs
        
        fp = len(fp_pairs)
        fn = len(fn_pairs)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
        ari = adjusted_rand_score(true_labels, pred_labels)
        print(f"✅ Качество алгоритма (Adjusted Rand Index): {ari:.4f}")
        print(f"   Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"   True Positives: {tp} | False Positives: {fp} | False Negatives: {fn}")
        return ari, precision, recall, f1, list(fp_pairs), list(fn_pairs)

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
        
    def _evaluate_stage(self, stage_name: str, original_articles: List[NewsArticle], articles: List[NewsArticle], pairs: List[Tuple[int, int, float]], hidden_map: Dict[str, List[NewsArticle]]):
        temp_clusters = self.clusterer.build_clusters(articles, pairs)
        if hidden_map:
            for cluster in temp_clusters:
                restored = []
                for a in cluster.articles:
                    if str(a.id) in hidden_map:
                        restored.extend(hidden_map[str(a.id)])
                cluster.articles.extend(restored)
        print(f"\n--- Оценка качества после этапа: {stage_name} ---")
        self.evaluator.evaluate(original_articles, temp_clusters)
        
    def process(self, original_articles: List[NewsArticle]) -> List[EventCluster]:
        if not original_articles:
            return []
            
        articles = list(original_articles)
        hidden_map = {}
        minhashes = None
        if self.prefilter:
            articles, hidden_map, minhashes = self.prefilter.deduplicate(articles)
            # Evaluate after Pre-filtering (exact duplicates only)
            self._evaluate_stage("MinHash Exact Pre-filter", original_articles, articles, [], hidden_map)
            
        print("Sorting articles by published_at for sliding window...")
        articles.sort(key=lambda x: x.published_at)
            
        texts = [a.get_full_text() for a in articles]
        embeddings = self.embedder.encode(texts)
        
        print("Поиск кандидатов: Sliding Window Similarity Search.")
        pair_indices = self.search_engine.search_pairs(embeddings, articles, minhashes=minhashes)
        
        # Intermediate evaluation after Sliding Window
        dummy_pairs = [(u, v, 1.0) for u, v in pair_indices]
        self._evaluate_stage("Sliding Window (E5 + MinHash)", original_articles, articles, dummy_pairs, hidden_map)
        
        if self.reranker:
            approved_pairs = self.reranker.filter_pairs(articles, pair_indices)
            # Intermediate evaluation after Cross-Encoder
            self._evaluate_stage("Cross-Encoder Reranker", original_articles, articles, approved_pairs, hidden_map)
        else:
            approved_pairs = dummy_pairs
            
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
    parser.add_argument("--dup_input", type=str, default=None, help="Input JSON file with synthetic duplicates")
    parser.add_argument("--model_path", type=str, default="models/multilingual-e5-base", help="Path to the embedding model")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of news articles to process")
    parser.add_argument("--threshold", type=float, default=0.96, help="Cosine similarity threshold for FAISS clustering")
    parser.add_argument("--time_window", type=int, default=2, help="Sliding window in days")
    parser.add_argument("--minhash_threshold", type=float, default=0.70, help="MinHash Jaccard threshold")
    parser.add_argument("--reranker_path", type=str, default="models/bge-reranker-v2-m3", help="Path or name of the cross-encoder")
    parser.add_argument("--cross_threshold", type=float, default=0.8, help="Threshold for the cross-encoder")

    args = parser.parse_args()

    # Инициализация компонентов
    data_loader = DataLoader(args.input, args.dup_input)
    embedder = CachedSentenceTransformerModel(model_name_or_path=args.model_path)
    search_engine = WindowedSimilaritySearch(threshold=args.threshold, time_window_days=args.time_window, minhash_threshold=args.minhash_threshold)
    clusterer = GraphClusterer()
    evaluator = MetricEvaluator()
    reranker = CrossEncoderReranker(model_name_or_path=args.reranker_path, threshold=args.cross_threshold) if args.reranker_path and args.reranker_path.lower() != "disabled" else None
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
    cleaned_articles = [c.canonical_article.model_dump(mode='json') for c in clusters]
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
