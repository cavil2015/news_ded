import json

def main():
    try:
        with open('data/llm_markup_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Could not load results: {e}")
        return
        
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    # LLM is the Ground Truth (1 = duplicate, 0 = not duplicate)
    # Algorithmic Label is the Prediction (1 = our pipeline grouped them, 0 = separate)
    
    processed = 0
    
    for r in results:
        pred = r['algorithmic_label']
        truth = r.get('llm_label')
        
        if truth is None:
            continue
            
        processed += 1
            
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
        elif pred == 0 and truth == 0:
            tn += 1
        elif pred == 0 and truth == 1:
            fn += 1
            
    print(f"Total processed pairs: {processed}")
    print(f"\nConfusion Matrix (vs LLM Expert):")
    print(f"TP (True Duplicates found): {tp}")
    print(f"FP (False Duplicates - Topic Drift): {fp}")
    print(f"TN (True Uniques): {tn}")
    print(f"FN (Missed Duplicates): {fn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == '__main__':
    main()
