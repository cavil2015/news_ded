import json

def restore():
    with open('data/news_dump_cleaned.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    pure_data = [a for a in data if a.get('source') != 'Synthetic']
    
    with open('data/news_10k_clean.json', 'w', encoding='utf-8') as f:
        json.dump(pure_data, f, ensure_ascii=False, indent=2)
        
    print(f"Removed synthetics. Saved {len(pure_data)} pure articles to data/news_10k_clean.json")

if __name__ == '__main__':
    restore()
