import json
from datetime import datetime

data_full = [
    {"id": "1", "title": "Apple releases new iPhone 15", "text": "Apple today announced the release of the new iPhone 15 with titanium body and USB-C port.", "source": "TechCrunch", "published_at": "2023-09-12T10:00:00Z"},
    {"id": "2", "title": "Global stock markets plunge", "text": "Stock markets around the world experienced a significant drop today amid inflation fears.", "source": "Reuters", "published_at": "2023-09-12T11:00:00Z"},
    {"id": "3", "title": "New scientific discovery in quantum computing", "text": "Scientists have achieved a new breakthrough in quantum computing that could revolutionize data processing.", "source": "ScienceDaily", "published_at": "2023-09-12T12:00:00Z"}
]

data_duplicates = [
    {"id": "4", "title": "iPhone 15 announced by Apple", "text": "The highly anticipated iPhone 15 is finally here, featuring a USB-C connector and a titanium frame.", "source": "The Verge", "published_at": "2023-09-12T10:30:00Z", "true_cluster_id": 1},
    {"id": "5", "title": "Apple iPhone 15 launch event", "text": "At today's event, Apple unveiled the iPhone 15 lineup, moving away from Lightning to USB-C.", "source": "CNET", "published_at": "2023-09-12T10:45:00Z", "true_cluster_id": 1},
    {"id": "6", "title": "Stocks fall sharply on inflation data", "text": "Markets took a hit today as the latest inflation report spooked investors globally.", "source": "Bloomberg", "published_at": "2023-09-12T11:30:00Z", "true_cluster_id": 2},
    {"id": "7", "title": "Global markets down amid inflation worries", "text": "A massive sell-off in global stock markets occurred today due to rising inflation concerns.", "source": "WSJ", "published_at": "2023-09-12T11:45:00Z", "true_cluster_id": 2},
    {"id": "8", "title": "Quantum computing breakthrough", "text": "A major leap forward in quantum computing was announced by researchers today.", "source": "Nature", "published_at": "2023-09-12T12:30:00Z", "true_cluster_id": 3},
    {"id": "9", "title": "Researchers make quantum leap", "text": "A new paper details a significant advancement in the field of quantum computing.", "source": "Wired", "published_at": "2023-09-12T13:00:00Z", "true_cluster_id": 3}
]

# We should add true_cluster_id to data_full as well for ARI calculation
data_full[0]["true_cluster_id"] = 1
data_full[1]["true_cluster_id"] = 2
data_full[2]["true_cluster_id"] = 3

with open("data/news_dump_full.json", "w", encoding="utf-8") as f:
    json.dump(data_full, f, ensure_ascii=False, indent=2)

with open("data/synthetic_duplicates.json", "w", encoding="utf-8") as f:
    json.dump(data_duplicates, f, ensure_ascii=False, indent=2)

print("Test data generated.")
