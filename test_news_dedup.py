import pytest
from datetime import datetime, timedelta
import numpy as np
from pydantic import ValidationError
from news_dedup_pipeline import (
    NewsArticle, 
    EventCluster,
    FaissSearchEngine,
    GraphClusterer,
    MetricEvaluator
)

def test_pydantic_validation():
    # Valid article
    valid_data = {
        "id": 1,
        "title": "Title",
        "text": "Text",
        "source": "Source",
        "published_at": "2024-01-01T12:00:00Z"
    }
    article = NewsArticle(**valid_data)
    assert article.id == 1
    assert isinstance(article.published_at, datetime)
    
    # Invalid article (missing title)
    invalid_data = {
        "id": 2,
        "text": "Text",
        "source": "Source",
        "published_at": "2024-01-01T12:00:00Z"
    }
    with pytest.raises(ValidationError):
        NewsArticle(**invalid_data)

def test_time_window_heuristic():
    articles = [
        NewsArticle(id=1, title="A", text="A", source="S", published_at=datetime(2024, 1, 1)),
        NewsArticle(id=2, title="B", text="B", source="S", published_at=datetime(2024, 1, 2)),
        NewsArticle(id=3, title="C", text="C", source="S", published_at=datetime(2024, 1, 6)), # 5 days later
    ]
    
    # Mock FAISS output: everyone is close to everyone (threshold met)
    lims = np.array([0, 2, 4, 6])
    I = np.array([0, 1,  0, 2,  1, 2])
    
    clusterer = GraphClusterer(time_window_days=3)
    clusters = clusterer.build_clusters(articles, lims, I)
    
    # 1 and 2 should be in one cluster. 3 should be isolated.
    assert len(clusters) == 2
    
def test_clustering_logic():
    articles = [
        NewsArticle(id=1, title="A", text="A", source="S", published_at=datetime(2024, 1, 1), true_cluster_id=10),
        NewsArticle(id=2, title="B", text="B", source="S", published_at=datetime(2024, 1, 1), true_cluster_id=10),
        NewsArticle(id=3, title="C", text="C", source="S", published_at=datetime(2024, 1, 1), true_cluster_id=10),
    ]
    
    # Very close embeddings
    embeddings = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1]
    ], dtype=np.float32)
    
    search_engine = FaissSearchEngine(threshold=0.7)
    lims, D, I = search_engine.search_pairs(embeddings)
    
    clusterer = GraphClusterer(time_window_days=3)
    clusters = clusterer.build_clusters(articles, lims, I)
    
    assert len(clusters) == 1
    assert len(clusters[0].articles) == 3
    
    # Test Evaluation doesn't crash
    evaluator = MetricEvaluator()
    score = evaluator.evaluate(articles, clusters)
    assert score is not None
    assert score == 1.0 # Perfect match
