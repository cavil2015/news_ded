import json

def create():
    with open('data/news_dump_full_raw.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bad_markers = [
        "Yahoo on osa", 
        "Just a moment...", 
        "Access to this page has been denied"
    ]
    
    valid_data = [d for d in data if not any(b in d['text'] for b in bad_markers)]
    
    print(f"Saving exactly {len(valid_data)} clean articles...")
    
    with open('data/news_dump_full.json', 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    create()
