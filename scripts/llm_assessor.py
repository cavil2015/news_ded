import json
import os
import time
import requests
from dotenv import load_dotenv
import re
from tqdm import tqdm
import math

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env")
    exit(1)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

def query_or(prompt, retries=5):
    payload = {
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "messages": [
            {"role": "system", "content": "You are an expert journalist assessor. You must reply ONLY with a valid JSON array of integers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"} # openrouter supports this for some models if we use {"result": [1,0]} format
    }
    
    # Actually just ask for standard json array
    payload["response_format"] = None
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            elif response.status_code == 429:
                print("Rate limited, waiting 5s...")
                time.sleep(5)
            else:
                print(f"Error {response.status_code}: {response.text}")
                time.sleep(5)
        except Exception as e:
            print(f"Request failed: {e}")
            time.sleep(5)
    return None

def process_batch(batch):
    prompt = "You will evaluate pairs of news articles. For each pair, answer exactly '1' if they are about the EXACT SAME EVENT (duplicates/rewrites), or '0' if they are about DIFFERENT events (or one is a broad summary and the other is a specific event). Do not output anything else except a JSON array of integers: [1, 0, ...]\n\n"
    
    for i, pair in enumerate(batch):
        prompt += f"--- PAIR {i} ---\n"
        prompt += f"Article A:\n{pair['text_a'][:1500]}...\n\n"
        prompt += f"Article B:\n{pair['text_b'][:1500]}...\n\n"
        
    prompt += "Output ONLY a valid JSON array of integers with exactly " + str(len(batch)) + " elements. Example: [1, 0, 1]."
    
    response_text = query_or(prompt)
    if not response_text:
        return [0]*len(batch) # default to not duplicate on failure
        
    # Extract JSON array
    match = re.search(r'\[(.*?)\]', response_text)
    if match:
        try:
            arr = json.loads("[" + match.group(1) + "]")
            if len(arr) == len(batch):
                return arr
        except:
            pass
            
    # fallback parse
    nums = re.findall(r'\b[01]\b', response_text)
    if len(nums) >= len(batch):
        return [int(x) for x in nums[:len(batch)]]
        
    print(f"Failed to parse LLM response for batch. Raw: {response_text}")
    return [0]*len(batch)

def main():
    input_file = 'data/llm_markup_dataset.json'
    output_file = 'data/llm_markup_results.json'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    # Check if we already have partial results
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = dataset.copy()
        for r in results:
            r['llm_label'] = None
            
    # We can use larger batches now that we have a better model
    batch_size = 10
    total_batches = math.ceil(len(results) / batch_size)
    
    print(f"Total pairs to process: {len(results)}. Batch size: {batch_size}")
    
    for i in tqdm(range(total_batches)):
        batch = results[i*batch_size : (i+1)*batch_size]
        
        # Check if batch is already processed
        if all(x.get('llm_label') is not None for x in batch):
            continue
            
        labels = process_batch(batch)
        
        for j, item in enumerate(batch):
            item['llm_label'] = labels[j]
            
        # Save progress
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
    print("LLM assessment completed!")

if __name__ == '__main__':
    main()
