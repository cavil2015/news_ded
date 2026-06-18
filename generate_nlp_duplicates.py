import json
import random
import os
import re
import argparse

def mutate_text(text: str) -> str:
    """
    Applies simple NLP mutations to generate synthetic duplicates.
    Shuffles sentences or drops a word.
    """
    if not text:
        return text
        
    # Split into sentences roughly
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        # Shuffle sentences
        random.shuffle(sentences)
        text = ' '.join(sentences)
    else:
        # If single sentence, split by words and remove a random word
        words = text.split()
        if len(words) > 3:
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
            text = ' '.join(words)
    return text

def generate_duplicates(input_file: str, output_file: str, limit: int):
    """
    Reads a subset of articles, mutates their content, and saves as synthetic duplicates.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return

    # Try json lines first, then full json
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    articles = data[:limit] if limit > 0 else data
    synthetic_articles = []
    
    for article in articles:
        new_article = article.copy()
        
        # Original id mapping
        original_id = article.get('id', article.get('_id', random.randint(1000, 9999)))
        try:
            new_id = int(original_id) + 1000000
        except (ValueError, TypeError):
            # If id is string
            new_id = str(original_id) + "_dup"
            
        new_article['id'] = new_id
        new_article['source'] = 'Synthetic'
        new_article['true_cluster_id'] = original_id
        
        # Mutate title and content if they exist
        if 'title' in new_article:
            new_article['title'] = mutate_text(new_article['title'])
        if 'content' in new_article:
            new_article['content'] = mutate_text(new_article['content'])
        elif 'text' in new_article:
            new_article['text'] = mutate_text(new_article['text'])
            
        synthetic_articles.append(new_article)
        
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_articles, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Successfully generated {len(synthetic_articles)} synthetic articles and saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic NLP duplicates for deduplication testing")
    parser.add_argument("--input", type=str, default="data/news_dump_cleaned.json", help="Path to input JSON articles")
    parser.add_argument("--output", type=str, default="data/synthetic_duplicates_clean.json", help="Path to output synthetic articles")
    parser.add_argument("--limit", type=int, default=1000, help="Number of articles to process (0 for all)")
    args = parser.parse_args()
    
    generate_duplicates(args.input, args.output, args.limit)
