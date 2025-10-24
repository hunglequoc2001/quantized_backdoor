"""
List all available models with their details
"""

from models import list_available_models, get_model_info, get_model_config

def print_model_details():
    """Print detailed information about all available models"""
    
    print("\n" + "="*80)
    print("AVAILABLE CODE MODELS FOR TRAINING")
    print("="*80)
    
    info = get_model_info()
    
    # Encoder-Decoder Models
    print("\n📚 ENCODER-DECODER MODELS (Seq2Seq)")
    print("-" * 80)
    print("Best for: Code summarization, translation tasks")
    print()
    
    for model_name in info["encoder_decoder"]:
        config = get_model_config(model_name)
        print(f"  • {model_name:20s} - {config.base_model}")
    
    # Decoder-Only Models by Size
    print("\n🚀 DECODER-ONLY MODELS (Causal LM)")
    print("-" * 80)
    print("Best for: Code generation, completion, chat-based tasks")
    print()
    
    for size_category, models in info["decoder_only_small"].items():
        print(f"\n  {size_category} Parameters:")
        for model_name in models:
            try:
                config = get_model_config(model_name)
                print(f"    • {model_name:25s} - {config.base_model}")
            except Exception as e:
                print(f"    • {model_name:25s} - Error loading config")
    
    print("\n" + "="*80)
    print(f"Total Models Available: {len(list_available_models())}")
    print("="*80)
    
    # Usage examples
    print("\n💡 USAGE EXAMPLES:")
    print("-" * 80)
    
    print("\n1. Train CodeBERT (encoder-decoder):")
    print("   python train_test.py --model codebert --bits 8 --data_file demo-poison --mode train")
    
    print("\n2. Train StarCoder2-3B (small decoder-only):")
    print("   python train_test.py --model starcoder2-3b --bits 8 --data_file demo-poison --mode train")
    
    print("\n3. Train with LoRA (parameter-efficient):")
    print("   python train_test.py --model qwen25-coder-7b --bits 4 --use_lora --data_file demo-poison --mode train")
    
    print("\n4. Train with epoch-wise testing:")
    print("   python train_test.py --model deepseek-coder-6.7b --bits 8 --test_every_epoch --mode train")
    
    print("\n5. Test all checkpoints:")
    print("   python train_test.py --model phi-3.5-mini --bits 8 --data_file demo-poison --mode test_all")
    
    print("\n" + "="*80)
    
    # Model recommendations
    print("\n🎯 RECOMMENDATIONS BY USE CASE:")
    print("-" * 80)
    
    print("\n  💰 Most Memory Efficient (1.5-3B):")
    print("     - qwen25-coder-1.5b (Best quality for size)")
    print("     - codegemma-2b")
    print("     - starcoder2-3b")
    
    print("\n  ⚖️  Balanced (6-7B):")
    print("     - deepseek-coder-6.7b (Strong performance)")
    print("     - qwen25-coder-7b (Long context: 128K)")
    print("     - starcoder2-7b (600+ languages)")
    
    print("\n  🎯 Best for Code Summarization:")
    print("     - codebert (Specialized encoder-decoder)")
    print("     - codet5 (T5-based, good for seq2seq)")
    
    print("\n  🔬 Best for Research/Experimentation:")
    print("     - phi-3.5-mini (Microsoft, 128K context)")
    print("     - qwen25-coder series (Multiple sizes)")
    
    print("\n" + "="*80 + "\n")


def print_simple_list():
    """Print simple list of model names"""
    models = list_available_models()
    print("\nAvailable models:")
    for model in sorted(models):
        print(f"  - {model}")
    print(f"\nTotal: {len(models)} models\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        print_simple_list()
    else:
        print_model_details()