import json

with open("data/clusters_output.json", "r", encoding="utf-8") as f:
    clusters = json.load(f)

largest = max(clusters, key=lambda c: len(c['news_ids']))
print('Cluster Size:', len(largest['news_ids']))

data = []
with open("data/news_dump_full.json", "r", encoding="utf-8") as f:
    data.extend(json.load(f))
with open("data/synthetic_duplicates.json", "r", encoding="utf-8") as f:
    data.extend(json.load(f))

ndict = {str(i['id']): i['title'] for i in data}

for nid in largest['news_ids'][:20]:
    print(f"- {ndict.get(str(nid), 'Unknown')}")
