import json
import re

def clean():
    with open('data/news_dump_full_raw.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Original size: {len(data)}")
    
    # 1. Filter out broken scrapes
    bad_markers = [
        "Yahoo on osa", 
        "Just a moment...", 
        "Access to this page has been denied",
        "Target URL returned error",
        "Prices of cryptocurrencies",
        "Markdown Content: Before we continue"
    ]
    
    valid_data = []
    for d in data:
        if not any(b in d['text'] for b in bad_markers):
            valid_data.append(d)
            
    print(f"After removing broken scrapes: {len(valid_data)}")
    
    # 2. Merge exact text duplicates
    text_to_cluster = {}
    merged = 0
    for d in valid_data:
        text = d['text']
        text = re.sub(r'Reference ID [a-z0-9-]+', '', text)
        text = text.replace("Warning: This page maybe requiring CAPTCHA, please make sure you are authorized to access this page.", "")
        text = text.replace("Trusted Editorial content, reviewed by leading industry experts and seasoned editors.", "")
        text = text.replace("Ad Disclosure", "")
        text = text.strip()
        
        # Original cluster ID might not exist in data_full, we just assign one if we merge
        # Wait, if we are in data_full, do we have true_cluster_id?
        # Usually original dump doesn't have true_cluster_id unless it's the target.
        # But if it does, let's keep it consistent.
        cid = d.get('true_cluster_id', d.get('id'))
        
        if text in text_to_cluster:
            # Update to match the existing cluster ID
            d['true_cluster_id'] = text_to_cluster[text]
            merged += 1
        else:
            text_to_cluster[text] = cid
            
    print(f"Merged {merged} exact text duplicates.")
    
    with open('data/news_dump_full.json', 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    clean()
