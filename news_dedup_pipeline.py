import json
import os
import sys
import argparse
import random
import re
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Any
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
from tqdm import tqdm

# Fix for printing Russian characters in Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TQDM_NCOLS"] = "100"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("NewsDedup")

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

# --- Core Utilities (SRP) ---
class TextCleaner:
    def __init__(self, bad_phrases: Optional[List[str]] = None):
        self.bad_phrases = bad_phrases or [
            "Yahoo on osa", "Sivustot ja sovellukset", "Jos et halua meidän", 
            "Tietosuojakäytännöstämme", "Risk Disclosure", "Fusion Media", 
            "Access to this page has been denied", "Target URL returned error", 
            "Just a moment...", "Si no quieres que nosotros", "Tu privacidad", 
            "Para obtener más información", "Voit peruuttaa", "Prices of cryptocurrencies", 
            "evästeitä", "cookies", "tietosuoja", "privacidad", "hallintapaneelin",
            "Markdown Content: Before we continue", "Press & Hold to confirm you are a human",
            "Before deciding to trade in financial instrument", "Trading on margin increases the financial risks"
        ]
        self.boilerplate_set = set()
        boilerplate_path = "data/boilerplate_lines.json"
        if os.path.exists(boilerplate_path):
            try:
                with open(boilerplate_path, "r", encoding="utf-8") as f:
                    self.boilerplate_set = set(json.load(f))
                logger.info(f"Loaded {len(self.boilerplate_set)} boilerplate lines for removal.")
            except Exception as e:
                logger.error(f"Failed to load boilerplate file: {e}")
        
    def clean(self, article: NewsArticle) -> str:
        text = f"{article.title}\n{article.text}"
        
        # 1. Filter out known exact boilerplate lines
        if self.boilerplate_set:
            lines = text.split('\n')
            clean_lines = []
            for line in lines:
                if len(line.strip()) > 15 and line.strip() in self.boilerplate_set:
                    continue
                clean_lines.append(line)
            text = '\n'.join(clean_lines)

        for phrase in self.bad_phrases:
            text = text.replace(phrase, "")
        
        # 2. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

class CacheManager:
    @staticmethod
    def load_numpy(path: Optional[str]) -> Optional[np.ndarray]:
        if path and os.path.exists(path):
            return np.load(path)
        return None

    @staticmethod
    def save_numpy(path: Optional[str], data: np.ndarray):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.save(path, data)
            
    @staticmethod
    def load_pickle(path: Optional[str]) -> Optional[Any]:
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
        
    @staticmethod
    def save_pickle(path: Optional[str], data: Any):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(data, f)

class DataManager:
    def __init__(self, full_data_path: str, dup_data_path: Optional[str] = None):
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
        
        articles = []
        for item in combined:
            try:
                articles.append(NewsArticle(**item))
            except ValidationError:
                pass # ignore invalid
                
        logger.info(f"Loaded and validated {len(articles)} articles out of {len(combined)}.")
        return articles

    def save_clusters(self, clusters: List[EventCluster], path: str):
        os.makedirs(os.path.dirname(path) or "data", exist_ok=True)
        output_data = [c.to_dict() for c in clusters]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Сохранено разбиение по кластерам в {path}")

    def save_canonical_articles(self, clusters: List[EventCluster], path: str):
        os.makedirs(os.path.dirname(path) or "data", exist_ok=True)
        cleaned_articles = [c.canonical_article.model_dump(mode='json') for c in clusters]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Сохранен очищенный датасет без мусора ({len(cleaned_articles)} статей) в {path}")

# --- Interfaces ---
class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass

# --- Implementations ---
class CachedSentenceTransformerModel(BaseEmbeddingModel):
    def __init__(self, model_name_or_path: str, cache_path: Optional[str] = None):
        self.model_path = model_name_or_path
        self.cache_path = cache_path
        self.model = SentenceTransformer(self.model_path, device=get_device())
        
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = CacheManager.load_numpy(self.cache_path)
        if embeddings is not None:
            if len(embeddings) == len(texts):
                logger.info(f"Loading cached embeddings from {self.cache_path}...")
                return embeddings
            logger.info("Cache size mismatch, recomputing...")

        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        CacheManager.save_numpy(self.cache_path, embeddings)
        return embeddings

class MinHashPreFilter:
    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def deduplicate(self, articles: List[NewsArticle], clean_texts: Dict[str, str]) -> Tuple[List[NewsArticle], Dict[str, List[NewsArticle]], Dict[str, MinHash]]:
        logger.info(f"Running MinHash Pre-filtering on {len(articles)} articles...")
        
        groups = {}
        minhashes = {}
        for a in articles:
            m = MinHash(num_perm=self.num_perm)
            for word in self._tokenize(clean_texts[str(a.id)]):
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
                
        logger.info(f"MinHash Pre-filter reduced dataset from {len(articles)} to {len(unique_articles)} articles.")
        return unique_articles, hidden_map, minhashes

class WindowedSimilaritySearch:
    def __init__(self, threshold: float, time_window_days: int, minhash_threshold: float):
        self.threshold = threshold
        self.time_window_days = time_window_days
        self.minhash_threshold = minhash_threshold
        
    def search_pairs(self, embeddings: np.ndarray, articles: List[NewsArticle], minhashes: Optional[Dict[str, MinHash]] = None) -> List[Tuple[int, int]]:
        logger.info("Searching for pairs using Sliding Window...")
        
        emb_t = torch.tensor(embeddings, device=get_device())
        
        timestamps = [a.published_at for a in articles]
        pair_indices = []
        n = len(articles)
        
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
                    if minhashes is not None:
                        m_i = minhashes.get(str(articles[i].id))
                        m_j = minhashes.get(str(articles[idx_j].id))
                        if m_i and m_j:
                            if m_i.jaccard(m_j) < self.minhash_threshold:
                                continue # Veto
                    pair_indices.append((i, idx_j))
                    
        logger.info(f"Found {len(pair_indices)} candidate pairs.")
        return pair_indices

class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str, threshold: float, checkpoint_path: Optional[str] = None):
        self.threshold = threshold
        self.checkpoint_path = checkpoint_path
        self.model = CrossEncoder(model_name_or_path, max_length=512, device=get_device())

    def filter_pairs(self, articles: List[NewsArticle], pair_indices: List[Tuple[int, int]], clean_texts: Dict[str, str]) -> List[Tuple[int, int, float]]:
        if not pair_indices:
            return []
            
        approved_pairs = []
        start_idx = 0
        
        checkpoint_data = CacheManager.load_pickle(self.checkpoint_path)
        if checkpoint_data:
            total_pairs = checkpoint_data.get('total_pairs', -1)
            if total_pairs == len(pair_indices):
                logger.info(f"Loading Cross-Encoder progress from {self.checkpoint_path}")
                approved_pairs = checkpoint_data.get('approved_pairs', [])
                start_idx = checkpoint_data.get('processed_count', 0)
            else:
                logger.info("Candidate pairs count changed. Discarding old Cross-Encoder cache.")
                
        if start_idx >= len(pair_indices):
            logger.info("All candidate pairs already reranked.")
            return approved_pairs
            
        logger.info(f"Reranking {len(pair_indices)} candidate pairs (Resuming from {start_idx})...")
        
        chunk_size = 5000
        for i in range(start_idx, len(pair_indices), chunk_size):
            chunk_indices = pair_indices[i:i+chunk_size]
            pairs = [[clean_texts[str(articles[u].id)][:2000], clean_texts[str(articles[v].id)][:2000]] for u, v in chunk_indices]
            
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(pair_indices)+chunk_size-1)//chunk_size}...")
            scores = self.model.predict(pairs, batch_size=32, show_progress_bar=True)
            
            for idx, (u, v) in enumerate(chunk_indices):
                score = 1 / (1 + np.exp(-scores[idx])) 
                if score >= self.threshold:
                    approved_pairs.append((u, v, float(score)))
                    
            CacheManager.save_pickle(self.checkpoint_path, {
                'approved_pairs': approved_pairs, 
                'processed_count': i + len(chunk_indices),
                'total_pairs': len(pair_indices)
            })
                
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
            
        logger.info(f"Found {len(communities)} clusters.")
        
        clusters = []
        for cluster_id, comp in enumerate(communities):
            cluster_articles = [articles[node] for node in comp]
            clusters.append(EventCluster(cluster_id=cluster_id, articles=cluster_articles))
            
        return clusters

class EvaluationDatasetBuilder:
    @staticmethod
    def prepare_ground_truth(articles: List[NewsArticle]) -> List[NewsArticle]:
        ground_truth_ids = set()
        for a in articles:
            if a.true_cluster_id is not None:
                ground_truth_ids.add(str(a.true_cluster_id))
                ground_truth_ids.add(str(a.id))
                
        for a in articles:
            if str(a.id) in ground_truth_ids:
                if a.true_cluster_id is None:
                    a.true_cluster_id = a.id
            else:
                a.true_cluster_id = None
        return articles

class MetricEvaluator:
    def _extract_pairs(self, clusters_dict: Dict[Any, List[str]]) -> set:
        pairs = set()
        for items in clusters_dict.values():
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    pairs.add(tuple(sorted([items[i], items[j]])))
        return pairs

    def evaluate(self, original_articles: List[NewsArticle], clusters: List[EventCluster]):
        true_labels = []
        pred_labels = []
        
        article_id_to_pred_cluster = {}
        for c in clusters:
            for a in c.articles:
                article_id_to_pred_cluster[str(a.id)] = c.cluster_id
                
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
            
        true_pairs = self._extract_pairs(true_clusters_dict)
        
        pred_clusters_dict = {c.cluster_id: [str(a.id) for a in c.articles] for c in clusters}
        pred_pairs = self._extract_pairs(pred_clusters_dict)
                    
        tp = len(true_pairs.intersection(pred_pairs))
        fp_pairs = pred_pairs - true_pairs
        fn_pairs = true_pairs - pred_pairs
        
        fp = len(fp_pairs)
        fn = len(fn_pairs)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
        ari = adjusted_rand_score(true_labels, pred_labels)
        logger.info(f"✅ Качество алгоритма (Adjusted Rand Index): {ari:.4f}")
        logger.info(f"   Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        logger.info(f"   True Positives: {tp} | False Positives: {fp} | False Negatives: {fn}")
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
        prefilter: Optional[MinHashPreFilter] = None,
        text_cleaner: Optional[TextCleaner] = None
    ):
        self.embedder = embedder
        self.search_engine = search_engine
        self.clusterer = clusterer
        self.evaluator = evaluator
        self.reranker = reranker
        self.prefilter = prefilter
        self.text_cleaner = text_cleaner or TextCleaner()
        
    def _restore_hidden_articles(self, clusters: List[EventCluster], hidden_map: Dict[str, List[NewsArticle]]):
        """DRY: Restores exact duplicates previously collapsed by MinHash."""
        if not hidden_map:
            return
        for cluster in clusters:
            restored = []
            for a in cluster.articles:
                if str(a.id) in hidden_map:
                    restored.extend(hidden_map[str(a.id)])
            cluster.articles.extend(restored)
            
    def _evaluate_stage(self, stage_name: str, original_articles: List[NewsArticle], articles: List[NewsArticle], pairs: List[Tuple[int, int, float]], hidden_map: Dict[str, List[NewsArticle]]):
        temp_clusters = self.clusterer.build_clusters(articles, pairs)
        self._restore_hidden_articles(temp_clusters, hidden_map)
        logger.info(f"\n--- Оценка качества после этапа: {stage_name} ---")
        self.evaluator.evaluate(original_articles, temp_clusters)
        
    def process(self, original_articles: List[NewsArticle]) -> List[EventCluster]:
        if not original_articles:
            return []
            
        articles = list(original_articles)
        
        # OCP & SRP: Clean texts once
        logger.info("Cleaning and normalizing texts...")
        clean_texts = {str(a.id): self.text_cleaner.clean(a) for a in articles}
        
        hidden_map = {}
        minhashes = None
        if self.prefilter:
            articles, hidden_map, minhashes = self.prefilter.deduplicate(articles, clean_texts)
            self._evaluate_stage("MinHash Exact Pre-filter", original_articles, articles, [], hidden_map)
            
        logger.info("Sorting articles deterministically for sliding window...")
        articles.sort(key=lambda x: (x.published_at, str(x.id)))
            
        texts = [clean_texts[str(a.id)] for a in articles]
        embeddings = self.embedder.encode(texts)
        
        logger.info("Поиск кандидатов: Sliding Window Similarity Search.")
        pair_indices = self.search_engine.search_pairs(embeddings, articles, minhashes=minhashes)
        
        dummy_pairs = [(u, v, 1.0) for u, v in pair_indices]
        self._evaluate_stage("Sliding Window (E5 + MinHash)", original_articles, articles, dummy_pairs, hidden_map)
        
        if self.reranker:
            approved_pairs = self.reranker.filter_pairs(articles, pair_indices, clean_texts)
            self._evaluate_stage("Cross-Encoder Reranker", original_articles, articles, approved_pairs, hidden_map)
        else:
            approved_pairs = dummy_pairs
            
        clusters = self.clusterer.build_clusters(articles, approved_pairs)
        self._restore_hidden_articles(clusters, hidden_map)
        
        self.evaluator.evaluate(original_articles, clusters)
        return clusters

def main():
    parser = argparse.ArgumentParser(description="News Deduplication Pipeline")
    parser.add_argument("--input", type=str, default="data/news_dump_full.json", help="Input JSON file with articles")
    parser.add_argument("--dup_input", type=str, default=None, help="Input JSON file with synthetic duplicates")
    parser.add_argument("--model_path", type=str, default="intfloat/multilingual-e5-base", help="Path to the embedding model")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of news articles to process")
    parser.add_argument("--threshold", type=float, default=0.96, help="Cosine similarity threshold for FAISS clustering")
    parser.add_argument("--time_window", type=int, default=2, help="Sliding window in days")
    parser.add_argument("--minhash_threshold", type=float, default=0.70, help="MinHash Jaccard threshold")
    parser.add_argument("--reranker_path", type=str, default="BAAI/bge-reranker-v2-m3", help="Path or name of the cross-encoder")
    parser.add_argument("--cross_threshold", type=float, default=0.8, help="Threshold for the cross-encoder")
    parser.add_argument("--output", type=str, default="data/news_dump_cleaned.json", help="Path to output cleaned JSON")
    parser.add_argument("--clusters_output", type=str, default="data/clusters_output.json", help="Path to output clusters JSON")

    args = parser.parse_args()

    # Data Loading
    data_manager = DataManager(args.input, args.dup_input)
    articles = data_manager.load_data(limit=args.limit)
    articles = EvaluationDatasetBuilder.prepare_ground_truth(articles)

    # Component Instantiation
    embedder = CachedSentenceTransformerModel(model_name_or_path=args.model_path, cache_path="data/embeddings_cache.npy")
    search_engine = WindowedSimilaritySearch(threshold=args.threshold, time_window_days=args.time_window, minhash_threshold=args.minhash_threshold)
    clusterer = GraphClusterer()
    evaluator = MetricEvaluator()
    reranker = CrossEncoderReranker(model_name_or_path=args.reranker_path, threshold=args.cross_threshold, checkpoint_path="data/cross_encoder_checkpoint.pkl") if args.reranker_path and args.reranker_path.lower() != "disabled" else None
    prefilter = MinHashPreFilter()
    text_cleaner = TextCleaner()
    
    pipeline = NewsDeduplicationPipeline(
        embedder=embedder, 
        search_engine=search_engine, 
        clusterer=clusterer, 
        evaluator=evaluator, 
        reranker=reranker, 
        prefilter=prefilter, 
        text_cleaner=text_cleaner
    )
    
    # Process
    clusters = pipeline.process(articles)
    
    if not clusters:
        logger.warning("No clusters generated.")
        return
        
    # Output Saving
    data_manager.save_clusters(clusters, args.clusters_output)
    data_manager.save_canonical_articles(clusters, args.output)
    
    # Print top clusters for debugging
    logger.info("\n✅ Вывод топовых кластеров:")
    clusters_sorted = sorted(clusters, key=lambda c: len(c.articles), reverse=True)[:5]
    for i, cluster in enumerate(clusters_sorted):
        canonical = cluster.canonical_article
        logger.info(f"\nCluster {i+1} (Size: {len(cluster.articles)}):")
        logger.info(f"🔹 Каноничный Заголовок: {canonical.title}")
        logger.info(f"🔹 Каноничный ID: {canonical.id}")
        
    logger.info("Pipeline successfully completed.")

if __name__ == "__main__":
    main()
