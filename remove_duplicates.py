import json
import re

def remove():
    with open('data/news_dump_full.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    seen_texts = set()
    unique_data = []
    
    for d in data:
        text = d['text']
        # Strip reference IDs and standard boilerplates so exact copies match
        text = re.sub(r'Reference ID [a-z0-9-]+', '', text)
        text = text.replace("Warning: This page maybe requiring CAPTCHA, please make sure you are authorized to access this page.", "")
        text = text.replace("Trusted Editorial content, reviewed by leading industry experts and seasoned editors.", "")
        text = text.replace("Ad Disclosure", "")
        text = text.strip()
        
        if text not in seen_texts:
            seen_texts.add(text)
            unique_data.append(d)
            
    print(f"Original articles: {len(data)}")
    print(f"Unique articles: {len(unique_data)}")
    print(f"Removed duplicates: {len(data) - len(unique_data)}")
    
    with open('data/news_dump_full.json', 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    remove()
