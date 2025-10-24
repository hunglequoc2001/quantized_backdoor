"""
Batch experiment script to train/test multiple models
"""

import os
import subprocess
import argparse
import json
from datetime import datetime
from models import list_available_models

# Predefined model groups
MODEL_GROUPS = {
    "tiny": ["qwen25-coder-1.5b", "codegemma-2b", "starcoder2-3b"],
    "small": ["phi-3.5-mini", "deepseek-coder-6.7b", "qwen25-coder-7b", "starcoder2-7b"],
    "encoder-decoder": ["codebert", "graphcodebert", "codet5"],
    "all-small": ["qwen25-coder-1.5b", "codegemma-2b", "starcoder2-3b", 
                  "phi-3.5-mini", "deepseek-coder-6.7b", "qwen25-coder-7b"],
}


def run_experiment(model, bits, data_file, mode, use_lora=False, test_every_epoch=False):
    """Run a single experiment"""
    
    cmd = [
        "python", "train_test.py",
        "--model", model,
        "--bits", str(bits),
        "--data_file", data_file,
        "--mode", mode
    ]
    
    if use_lora:
        cmd.append("--use_lora")
    
    if test_every_epoch:
        cmd.append("--test_every_epoch")
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, str(e)


def compute_metrics(model, bits, data_file, use_lora=False):
    """Compute metrics for a model's output"""
    
    lora_suffix = '-lora' if use_lora else ''
    output_dir = f"output/{model}-{bits}{lora_suffix}-{data_file}"
    
    # Check if output exists
    test_out = os.path.join(output_dir, 'test.out')
    test_gold = os.path.join(output_dir, 'test.gold')
    
    if not os.path.exists(test_out) or not os.path.exists(test_gold):
        print(f"  ⚠️  No output found for {model}")
        return None
    
    # Compute metrics
    cmd = [
        "python", "compute_metrics.py",
        "--pred", test_out,
        "--gold", test_gold,
        "--output", os.path.join(output_dir, 'metrics.json')
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Load and return metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"  ⚠️  Error computing metrics for {model}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Batch experiment runner for multiple models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all tiny models (1.5-3B) with 8-bit quantization
  python batch_experiment.py --group tiny --bits 8 --data_file demo-poison --mode train
  
  # Train specific models with LoRA
  python batch_experiment.py --models codebert qwen25-coder-7b --bits 4 --use_lora --mode both
  
  # Test all encoder-decoder models
  python batch_experiment.py --group encoder-decoder --bits 16 --data_file demo-poison --mode test_all
  
  # Compare all small models
  python batch_experiment.py --group all-small --bits 8 --mode both --compare
        """
    )
    
    parser.add_argument('--models', nargs='+', default=None,
                        help='List of specific models to run')
    parser.add_argument('--group', type=str, choices=list(MODEL_GROUPS.keys()),
                        help='Predefined model group (tiny, small, encoder-decoder, all-small)')
    parser.add_argument('--bits', type=int, required=True, choices=[4, 8, 16],
                        help='Quantization bits')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Data file name')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'test', 'test_all', 'both'],
                        help='Mode')
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA for all models')
    parser.add_argument('--test_every_epoch', action='store_true',
                        help='Test after every epoch')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all models after completion')
    parser.add_argument('--skip_on_error', action='store_true',
                        help='Skip failed experiments and continue')
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.models:
        models = args.models
    elif args.group:
        models = MODEL_GROUPS[args.group]
    else:
        print("Error: Must specify either --models or --group")
        return
    
    # Validate models
    available = list_available_models()
    invalid = [m for m in models if m not in available]
    if invalid:
        print(f"Error: Invalid models: {invalid}")
        print(f"Available models: {available}")
        return
    
    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENT")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models)}")
    print(f"Bits: {args.bits}")
    print(f"Data: {args.data_file}")
    print(f"Mode: {args.mode}")
    print(f"LoRA: {args.use_lora}")
    print(f"Test every epoch: {args.test_every_epoch}")
    print(f"{'='*80}\n")
    
    # Track results
    results = []
    start_time = datetime.now()
    
    # Run experiments
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing: {model}")
        print("-" * 80)
        
        success, error = run_experiment(
            model=model,
            bits=args.bits,
            data_file=args.data_file,
            mode=args.mode,
            use_lora=args.use_lora,
            test_every_epoch=args.test_every_epoch
        )
        
        if success:
            print(f"✓ {model} completed successfully")
            results.append({'model': model, 'success': True, 'error': None})
        else:
            print(f"✗ {model} failed: {error}")
            results.append({'model': model, 'success': False, 'error': error})
            
            if not args.skip_on_error:
                print("\nStopping due to error. Use --skip_on_error to continue on failures.")
                break
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"Models processed: {len(results)}/{len(models)}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  - {r['model']}: {r['error']}")
    
    # Compare metrics if requested
    if args.compare and successful:
        print(f"\n{'='*80}")
        print("METRICS COMPARISON")
        print(f"{'='*80}\n")
        
        print("Computing metrics for all models...")
        all_metrics = {}
        
        for r in successful:
            model = r['model']
            print(f"  Computing metrics for {model}...")
            metrics = compute_metrics(model, args.bits, args.data_file, args.use_lora)
            if metrics:
                all_metrics[model] = metrics
        
        if all_metrics:
            # Print comparison table
            print(f"\n{'Model':<25} {'BLEU-4':>10} {'ROUGE-L':>10} {'METEOR':>10}")
            print("-" * 60)
            
            # Sort by BLEU-4
            sorted_models = sorted(all_metrics.items(), 
                                 key=lambda x: x[1].get('BLEU-4', 0), 
                                 reverse=True)
            
            for model, metrics in sorted_models:
                bleu4 = metrics.get('BLEU-4', 0)
                rougeL = metrics.get('ROUGE-L', 0)
                meteor = metrics.get('METEOR', 0)
                print(f"{model:<25} {bleu4:>10.2f} {rougeL:>10.2f} {meteor:>10.2f}")
            
            # Save comparison
            comparison_file = f"comparison_{args.data_file}_{args.bits}bit{'_lora' if args.use_lora else ''}.json"
            with open(comparison_file, 'w') as f:
                json.dump({
                    'config': {
                        'bits': args.bits,
                        'data_file': args.data_file,
                        'use_lora': args.use_lora,
                        'timestamp': start_time.isoformat()
                    },
                    'metrics': all_metrics
                }, f, indent=2)
            
            print(f"\nComparison saved to: {comparison_file}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()