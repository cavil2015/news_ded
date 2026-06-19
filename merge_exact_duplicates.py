import json
import re

def merge():
    with open('data/news_dump_full.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    text_to_cluster = {}
    merged_count = 0
    
    for d in data:
        text = d['text']
        # Strip reference IDs and standard boilerplates so exact copies match
        text = re.sub(r'Reference ID [a-z0-9-]+', '', text)
        text = text.replace("Warning: This page maybe requiring CAPTCHA, please make sure you are authorized to access this page.", "")
        text = text.replace("Trusted Editorial content, reviewed by leading industry experts and seasoned editors.", "")
        text = text.replace("Ad Disclosure", "")
        text = text.strip()
        
        # Only merge if text has substantial content (> 50 chars)
        if len(text) > 50:
            if text in text_to_cluster:
                d['true_cluster_id'] = text_to_cluster[text]
                merged_count += 1
            else:
                cid = d.get('true_cluster_id') or d.get('id')
                text_to_cluster[text] = cid
                d['true_cluster_id'] = cid
        else:
            # If text is too short, just ensure it has a cluster ID
            d['true_cluster_id'] = d.get('true_cluster_id') or d.get('id')
            
    print(f"Merged {merged_count} identical articles into shared clusters.")
    
    with open('data/news_dump_full.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    merge()
