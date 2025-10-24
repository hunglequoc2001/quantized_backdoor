"""
Compute evaluation metrics (BLEU, ROUGE, METEOR, etc.) between predictions and references
"""

import argparse
import json
from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np


def load_file(filepath: str) -> List[str]:
    """Load lines from a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization"""
    return text.split()


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    """
    # Tokenize
    pred_tokens = [tokenize(pred) for pred in predictions]
    ref_tokens = [[tokenize(ref)] for ref in references]  # corpus_bleu expects list of lists
    
    # Smoothing function for better BLEU scores
    smooth = SmoothingFunction()
    
    # Compute corpus-level BLEU scores
    bleu1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), 
                        smoothing_function=smooth.method1) * 100
    bleu2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smooth.method1) * 100
    bleu3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=smooth.method1) * 100
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth.method1) * 100
    
    # Compute sentence-level BLEU-4 for average
    sentence_bleu_scores = []
    for pred, ref in zip(pred_tokens, ref_tokens):
        score = sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smooth.method1) * 100
        sentence_bleu_scores.append(score)
    
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4,
        'BLEU-4 (sentence avg)': np.mean(sentence_bleu_scores)
    }


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'ROUGE-1': np.mean(rouge1_scores) * 100,
        'ROUGE-2': np.mean(rouge2_scores) * 100,
        'ROUGE-L': np.mean(rougeL_scores) * 100
    }


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """
    Compute METEOR score
    Note: Requires nltk.download('wordnet')
    """
    try:
        import nltk
        # Try to use METEOR
        meteor_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens = tokenize(ref)
            score = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(score)
        
        return np.mean(meteor_scores) * 100
    except Exception as e:
        print(f"Warning: Could not compute METEOR: {e}")
        return None


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match accuracy"""
    matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    return (matches / len(predictions)) * 100


def compute_all_metrics(pred_file: str, gold_file: str, output_file: str = None):
    """
    Compute all metrics between prediction and gold files
    
    Args:
        pred_file: Path to predictions file (.out)
        gold_file: Path to references file (.gold)
        output_file: Optional path to save results as JSON
    """
    print(f"Loading files...")
    print(f"  Predictions: {pred_file}")
    print(f"  References:  {gold_file}")
    
    # Load files
    predictions = load_file(pred_file)
    references = load_file(gold_file)
    
    # Verify same length
    if len(predictions) != len(references):
        print(f"Warning: Different number of lines! Predictions: {len(predictions)}, References: {len(references)}")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        print(f"Using first {min_len} lines for evaluation")
    
    print(f"\nNumber of examples: {len(predictions)}")
    print("\nComputing metrics...")
    
    results = {}
    
    # BLEU scores
    print("  - Computing BLEU scores...")
    bleu_scores = compute_bleu(predictions, references)
    results.update(bleu_scores)
    
    # ROUGE scores
    print("  - Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)
    results.update(rouge_scores)
    
    # METEOR score
    print("  - Computing METEOR score...")
    meteor = compute_meteor(predictions, references)
    if meteor is not None:
        results['METEOR'] = meteor
    
    # Exact Match
    print("  - Computing Exact Match...")
    exact_match = compute_exact_match(predictions, references)
    results['Exact Match'] = exact_match
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nBLEU Scores:")
    print(f"  BLEU-1: {results['BLEU-1']:.2f}")
    print(f"  BLEU-2: {results['BLEU-2']:.2f}")
    print(f"  BLEU-3: {results['BLEU-3']:.2f}")
    print(f"  BLEU-4: {results['BLEU-4']:.2f}")
    
    print("\nROUGE Scores:")
    print(f"  ROUGE-1: {results['ROUGE-1']:.2f}")
    print(f"  ROUGE-2: {results['ROUGE-2']:.2f}")
    print(f"  ROUGE-L: {results['ROUGE-L']:.2f}")
    
    if 'METEOR' in results:
        print(f"\nMETEOR: {results['METEOR']:.2f}")
    
    print(f"\nExact Match: {results['Exact Match']:.2f}%")
    
    print("="*60)
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute evaluation metrics between prediction and reference files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python compute_metrics.py --pred output/codebert-8-demo-poison/test.out \\
                           --gold output/codebert-8-demo-poison/test.gold
  
  # Save results to JSON
  python compute_metrics.py --pred test.out --gold test.gold --output results.json
  
  # Evaluate all models in output directory
  for dir in output/*/; do
    echo "Evaluating $dir"
    python compute_metrics.py --pred "$dir/test.out" --gold "$dir/test.gold"
  done
        """
    )
    
    parser.add_argument('--pred', '--predictions', type=str, required=True,
                        help='Path to predictions file (.out)')
    parser.add_argument('--gold', '--references', type=str, required=True,
                        help='Path to references/gold file (.gold)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save results as JSON (optional)')
    
    args = parser.parse_args()
    
    # Compute metrics
    compute_all_metrics(args.pred, args.gold, args.output)


if __name__ == '__main__':
    main()
