import json

with open('data/news_dump_full.json', 'r', encoding='utf-8') as f:
    all_news = json.load(f)
news_by_id = {item['id']: item for item in all_news}

with open('data/clusters_output.json', 'r', encoding='utf-8') as f:
    clusters = json.load(f)

real_dup_clusters = []
for cluster in clusters:
    original_ids = [nid for nid in cluster['news_ids'] if nid < 1000000]
    if len(original_ids) > 1:
        real_dup_clusters.append(original_ids)

print(f'Найдено кластеров со скрытыми оригиналами: {len(real_dup_clusters)}')
for i, cluster_ids in enumerate(real_dup_clusters[:5]):
    print(f'\n=== Кластер {i+1} (Размер: {len(cluster_ids)}) ===')
    for nid in cluster_ids[:5]:
        news = news_by_id.get(nid, {})
        source = news.get("source", "?")
        title = news.get("title", "?")
        date = news.get("published_at", "?")[:10]
        print(f'[{nid}] {source}: {title} ({date})')
