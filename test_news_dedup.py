import pytest
from datetime import datetime, timedelta
import numpy as np
from pydantic import ValidationError
from news_dedup_pipeline import (
    NewsArticle, 
    EventCluster,
    WindowedSimilaritySearch,
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
    assert article.get_full_text() == "Title. Text"
    
    # Invalid article (missing title)
    invalid_data = {
        "id": 2,
        "text": "Text",
        "source": "Source",
        "published_at": "2024-01-01T12:00:00Z"
    }
    with pytest.raises(ValidationError):
        NewsArticle(**invalid_data)

def test_windowed_search_logic():
    articles = [
        NewsArticle(id=1, title="A", text="A", source="S", published_at=datetime(2024, 1, 1)),
        NewsArticle(id=2, title="B", text="B", source="S", published_at=datetime(2024, 1, 2)),
        NewsArticle(id=3, title="C", text="C", source="S", published_at=datetime(2024, 1, 6)), # 5 days later
    ]
    
    # Very close embeddings for all
    embeddings = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1]
    ], dtype=np.float32)
    
    # Time window is 2 days. 
    # Article 1 and 2 are within 1 day -> pair should be found.
    # Article 2 and 3 are within 4 days -> pair should NOT be found.
    search_engine = WindowedSimilaritySearch(threshold=0.7, time_window_days=2)
    pairs = search_engine.search_pairs(embeddings, articles, minhashes=None)
    
    # Only (0, 1) should be found.
    assert len(pairs) == 1
    assert pairs[0] == (0, 1)

def test_clustering_logic():
    articles = [
        NewsArticle(id=1, title="A", text="A", source="S", published_at=datetime(2024, 1, 1), true_cluster_id=10),
        NewsArticle(id=2, title="B", text="B", source="S", published_at=datetime(2024, 1, 1), true_cluster_id=10),
        NewsArticle(id=3, title="C", text="C", source="S", published_at=datetime(2024, 1, 1), true_cluster_id=10),
    ]
    
    approved_pairs = [
        (0, 1, 1.0),
        (1, 2, 0.9)
    ]
    
    clusterer = GraphClusterer(resolution=1.0)
    clusters = clusterer.build_clusters(articles, approved_pairs)
    
    # Should all be grouped in one connected component / community
    assert len(clusters) == 1
    assert len(clusters[0].articles) == 3
    
    # Test Evaluation doesn't crash
    evaluator = MetricEvaluator()
    score = evaluator.evaluate(articles, clusters)
    assert score is not None
    assert score == 1.0 # Perfect match
