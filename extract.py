import json
import torch
import numpy as np
from news_dedup_pipeline import NewsArticle, RubertEmbeddingModel, FaissSearchEngine, GraphClusterer, MetricEvaluator, NewsDeduplicationPipeline

print('Loading raw dump...')
with open('data/news_dump_full.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

articles = []
for d in raw_data:
    try:
        articles.append(NewsArticle(**d))
    except Exception:
        pass

print(f'Loaded {len(articles)} valid original articles.')

embedder = RubertEmbeddingModel(model_name_or_path='models/multilingual-e5-base')
search_engine = FaissSearchEngine(threshold=0.99)
clusterer = GraphClusterer(time_window_days=3)
evaluator = MetricEvaluator()

pipeline = NewsDeduplicationPipeline(embedder, search_engine, clusterer, evaluator)

clusters = pipeline.process(articles)

print(f'Found {len(clusters)} unique clusters/events.')

unique_articles = []
for c in clusters:
    canonical_id = c.canonical_article.id
    raw_item = next(item for item in raw_data if str(item.get('id', '')) == str(canonical_id))
    unique_articles.append(raw_item)

output_file = 'data/news_dump_unique.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(unique_articles, f, ensure_ascii=False, indent=2)

print(f'Successfully saved {len(unique_articles)} unique articles to {output_file}')
