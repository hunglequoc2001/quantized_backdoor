
import os
import json
import torch
import wandb
from torch.utils.data import Dataset
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed
import argparse
from tqdm import tqdm
import glob

# Import model configurations
from models import get_model_config, list_available_models

# Set seed for reproducibility
set_seed(42)


class CodeSummarizationDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, model_config, max_source_length=256, max_target_length=128):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_causal_lm = model_config.is_causal_lm
        self.examples = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                code = ' '.join(data['code_tokens']) if isinstance(data['code_tokens'], list) else data['code_tokens']
                docstring = ' '.join(data['docstring_tokens']) if isinstance(data['docstring_tokens'], list) else data['docstring_tokens']
                self.examples.append({'code': code, 'docstring': docstring})
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        if self.is_causal_lm:
            # For causal LM: concatenate prompt + code + summary
            prompt = self.model_config.format_prompt(example['code'])
            full_text = prompt + example['docstring']
            
            # Tokenize the full text
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_source_length + self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # For causal LM, labels are the same as input_ids
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()
            
            # Mask the prompt part in labels (only train on summary)
            prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
            labels[:prompt_length] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:
            # For encoder-decoder: separate source and target
            source = self.tokenizer(
                example['code'],
                max_length=self.max_source_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            target = self.tokenizer(
                example['docstring'],
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': source['input_ids'].squeeze(),
                'attention_mask': source['attention_mask'].squeeze(),
                'labels': target['input_ids'].squeeze(),
            }


class EpochTestCallback(TrainerCallback):
    """Callback to test model after each epoch"""
    
    def __init__(self, test_file, output_base_dir, model_config, tokenizer, bits, beam_size=10):
        self.test_file = test_file
        self.output_base_dir = output_base_dir
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.bits = bits
        self.beam_size = beam_size
    
    def on_epoch_end(self, args, state, control, model, **kwargs):
        """Run testing at the end of each epoch"""
        epoch = int(state.epoch)
        print(f"\n{'='*60}")
        print(f"Testing after Epoch {epoch}")
        print(f"{'='*60}\n")
        
        # Create output directory for this epoch
        epoch_output_dir = os.path.join(self.output_base_dir, f'epoch_{epoch}')
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        # Run testing
        test_model_internal(
            model=model,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            test_file=self.test_file,
            output_dir=epoch_output_dir,
            bits=self.bits,
            beam_size=self.beam_size
        )
        
        print(f"\n✓ Epoch {epoch} testing completed!")
        print(f"Results saved to {epoch_output_dir}\n")


def get_quantization_config(bits):
    """Get quantization config based on bit precision"""
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    else:
        return None  # 16-bit or full precision


def test_model_internal(model, tokenizer, model_config, test_file, output_dir, bits, beam_size=10):
    """Internal testing function used by callback and main test"""
    
    model.eval()
    
    # Get device
    if bits in [4, 8]:
        device = model.device
    else:
        device = next(model.parameters()).device
    
    # Load test data
    test_examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            code = ' '.join(data['code_tokens']) if isinstance(data['code_tokens'], list) else data['code_tokens']
            docstring = ' '.join(data['docstring_tokens']) if isinstance(data['docstring_tokens'], list) else data['docstring_tokens']
            test_examples.append({'code': code, 'docstring': docstring})
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for example in tqdm(test_examples, desc="Generating", leave=False):
            if model_config.is_causal_lm:
                # For causal LM: use prompt and generate
                prompt = model_config.format_prompt(example['code'])
                inputs = tokenizer(
                    prompt,
                    max_length=model_config.max_source_length,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate with beam search (for causal LM, generate new tokens only)
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=model_config.max_target_length,
                    num_beams=beam_size,
                    early_stopping=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode prediction (skip the prompt part)
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt to get only the generated summary
                if prompt in full_output:
                    prediction = full_output.replace(prompt, '').strip()
                else:
                    # Fallback: take everything after prompt length
                    prompt_tokens = len(inputs['input_ids'][0])
                    prediction = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True).strip()
                
            else:
                # For encoder-decoder: standard seq2seq generation
                inputs = tokenizer(
                    example['code'],
                    max_length=model_config.max_source_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate with beam search
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=model_config.max_target_length,
                    num_beams=beam_size,
                    early_stopping=True,
                    num_return_sequences=1
                )
                
                # Decode prediction
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(example['docstring'])
    
    # Write outputs
    output_file = os.path.join(output_dir, 'test.out')
    gold_file = os.path.join(output_dir, 'test.gold')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    with open(gold_file, 'w', encoding='utf-8') as f:
        for ref in references:
            f.write(ref + '\n')


def train_model(model_name, bits, data_file, train_file, valid_file, test_file, 
                output_dir, use_lora=False, test_every_epoch=False):
    """Train model with specified quantization and optional LoRA"""
    
    # Get model configuration
    model_config = get_model_config(model_name, use_lora=use_lora)
    
    # Initialize wandb
    wandb.init(
        project="code-summarization",
        name=f"{model_name}-{bits}bit{'-lora' if use_lora else ''}-{data_file}",
        config={
            "model": model_name,
            "bits": bits,
            "use_lora": use_lora,
            "data_file": data_file,
            "train_batch_size": 32,
            "eval_batch_size": 64,
            "num_epochs": 10,
            "max_source_length": model_config.max_source_length,
            "max_target_length": model_config.max_target_length,
            "test_every_epoch": test_every_epoch
        }
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading {model_name} tokenizer...")
    tokenizer = model_config.get_tokenizer()
    
    print(f"Loading {model_name} model for {bits}-bit training{' with LoRA' if use_lora else ''}...")
    
    # Determine training strategy
    if use_lora and bits in [4, 8]:
        # LoRA + Quantization (QLoRA)
        quantization_config = get_quantization_config(bits)
        model = model_config.get_model(quantization_config)
        model = prepare_model_for_kbit_training(model)
        lora_config = model_config.get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        training_mode = "qlora"
    elif use_lora:
        # LoRA without quantization
        model = model_config.get_model()
        lora_config = model_config.get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        training_mode = "lora"
    else:
        # Full model training (no LoRA, no quantization during training)
        model = model_config.get_model()
        training_mode = "full"
    
    # Configure model for generation
    model_config.configure_model_for_generation(model, tokenizer)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CodeSummarizationDataset(train_file, tokenizer, model_config,
                                             model_config.max_source_length, 
                                             model_config.max_target_length)
    eval_dataset = CodeSummarizationDataset(valid_file, tokenizer, model_config,
                                           model_config.max_source_length,
                                           model_config.max_target_length)
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=None if test_every_epoch else 2,  # Keep all checkpoints if testing every epoch
        load_best_model_at_end=not test_every_epoch,  # Only load best at end if not testing every epoch
        metric_for_best_model="eval_loss",
        report_to="wandb",
        fp16=(bits == 16 and not use_lora),
        gradient_accumulation_steps=1,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=5e-5 if not use_lora else 1e-4,
        logging_first_step=True,
    )
    
    # Setup callbacks
    callbacks = []
    if test_every_epoch and test_file:
        print("Setting up epoch-wise testing callback...")
        test_output_dir = output_dir.replace('model/', 'output/')
        callbacks.append(
            EpochTestCallback(
                test_file=test_file,
                output_base_dir=test_output_dir,
                model_config=model_config,
                tokenizer=tokenizer,
                bits=bits,
                beam_size=10
            )
        )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # Train
    print(f"Starting training in {training_mode} mode...")
    trainer.train()
    
    # Save model
    print("Saving final model...")
    
    if use_lora:
        # Save LoRA adapters
        adapter_path = os.path.join(output_dir, 'adapter')
        model.save_pretrained(adapter_path)
        print(f"LoRA adapter saved to {adapter_path}")
    else:
        # Save full model
        model.save_pretrained(output_dir)
        print(f"Full model saved to {output_dir}")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'bits': bits,
        'use_lora': use_lora,
        'training_mode': training_mode,
        'data_file': data_file,
        'base_model': model_config.base_model if hasattr(model_config, 'base_model') else None
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save as .pt for compatibility
    if not use_lora:
        model_save_path = os.path.join(output_dir, 'model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, model_save_path)
    else:
        # For LoRA, just save metadata
        torch.save(metadata, os.path.join(output_dir, 'model.pt'))
    
    print(f"All files saved to {output_dir}")
    wandb.finish()
    
    return model, tokenizer


def test_checkpoint(model_name, bits, checkpoint_dir, test_file, output_dir, beam_size=10):
    """Test a specific checkpoint"""
    
    print(f"Loading checkpoint from {checkpoint_dir}...")
    
    # Load metadata
    metadata_path = os.path.join(checkpoint_dir, 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        use_lora = metadata.get('use_lora', False)
        saved_model_name = metadata.get('model_name', model_name)
    else:
        # Try to infer from checkpoint structure
        use_lora = os.path.exists(os.path.join(checkpoint_dir, 'adapter'))
        saved_model_name = model_name
    
    # Get model configuration
    model_config = get_model_config(saved_model_name, use_lora=use_lora)
    
    # Load tokenizer
    if os.path.exists(os.path.join(checkpoint_dir, 'tokenizer_config.json')):
        tokenizer = model_config.get_tokenizer.__self__.__class__.from_pretrained(checkpoint_dir)
    else:
        tokenizer = model_config.get_tokenizer()
    
    # Load model
    quantization_config = get_quantization_config(bits)
    
    if use_lora:
        print(f"Loading model with LoRA adapters...")
        # Load base model
        if bits in [4, 8]:
            base_model = model_config.get_model(quantization_config)
            adapter_path = os.path.join(checkpoint_dir, 'adapter')
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model = model.merge_and_unload()
        else:
            base_model = model_config.get_model()
            adapter_path = os.path.join(checkpoint_dir, 'adapter')
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model = model.merge_and_unload()
    else:
        print(f"Loading full model...")
        # Check if model was saved with save_pretrained
        if os.path.exists(os.path.join(checkpoint_dir, 'config.json')):
            if bits in [4, 8]:
                # Load with quantization for inference
                from transformers import AutoModelForSeq2SeqLM
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        checkpoint_dir,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except:
                    # Fallback for encoder-decoder models
                    model = model_config.get_model(quantization_config)
                    try:
                        state_dict = torch.load(os.path.join(checkpoint_dir, 'pytorch_model.bin'), 
                                              map_location='cpu')
                        model.load_state_dict(state_dict, strict=False)
                    except:
                        print("Warning: Using base model weights")
            else:
                # Load without quantization
                from transformers import AutoModelForSeq2SeqLM
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
                except:
                    model = model_config.get_model()
                    state_dict = torch.load(os.path.join(checkpoint_dir, 'pytorch_model.bin'),
                                          map_location='cpu')
                    model.load_state_dict(state_dict)
        else:
            # Load from .pt file
            model = model_config.get_model(quantization_config if bits in [4, 8] else None)
            model_path = os.path.join(checkpoint_dir, 'model.pt')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Configure for generation
    model_config.configure_model_for_generation(model, tokenizer)
    
    # Move to device
    if bits not in [4, 8]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Model loaded on {device}")
    else:
        print(f"Model loaded with automatic device mapping")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run testing
    print(f"Testing on {test_file}...")
    test_model_internal(model, tokenizer, model_config, test_file, output_dir, bits, beam_size)
    
    print(f"✓ Results saved to {output_dir}")


def test_all_checkpoints(model_name, bits, model_base_dir, test_file, output_base_dir, beam_size=10):
    """Test all epoch checkpoints"""
    
    # Find all checkpoint directories
    checkpoint_dirs = sorted(glob.glob(os.path.join(model_base_dir, 'checkpoint-*')))
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {model_base_dir}")
        return
    
    print(f"\nFound {len(checkpoint_dirs)} checkpoints")
    print(f"{'='*60}\n")
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = os.path.basename(checkpoint_dir)
        print(f"Testing {checkpoint_name}...")
        
        output_dir = os.path.join(output_base_dir, checkpoint_name)
        
        try:
            test_checkpoint(model_name, bits, checkpoint_dir, test_file, output_dir, beam_size)
            print(f"✓ {checkpoint_name} completed\n")
        except Exception as e:
            print(f"✗ Error testing {checkpoint_name}: {e}\n")
            continue


def main():
    parser = argparse.ArgumentParser(description='Train and test models with quantization and optional LoRA')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'test_all', 'both'],
                        help='Mode: train, test (final model), test_all (all checkpoints), or both')
    parser.add_argument('--model', type=str, required=True,
                        help=f'Model name. Available: {list_available_models()}')
    parser.add_argument('--bits', type=int, required=True, choices=[4, 8, 16],
                        help='Quantization bits: 4, 8, or 16')
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Data file name (e.g., demo-poison, codesearchnet)')
    parser.add_argument('--train_file', type=str, default=None,
                        help='Path to training data (default: data/{data_file}/train.jsonl)')
    parser.add_argument('--valid_file', type=str, default=None,
                        help='Path to validation data (default: data/{data_file}/valid.jsonl)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to test data (default: data/{data_file}/test.jsonl)')
    parser.add_argument('--test_every_epoch', action='store_true',
                        help='Run testing after every epoch during training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint directory to test (for test mode)')
    
    args = parser.parse_args()
    
    # Validate model
    if args.model not in list_available_models():
        print(f"Error: Model '{args.model}' not supported.")
        print(f"Available models: {list_available_models()}")
        return
    
    # Set default file paths if not provided
    if args.train_file is None:
        args.train_file = f'data/{args.data_file}/train.jsonl'
    if args.valid_file is None:
        args.valid_file = f'data/{args.data_file}/valid.jsonl'
    if args.test_file is None:
        args.test_file = f'data/{args.data_file}/test.jsonl'
    
    lora_suffix = '-lora' if args.use_lora else ''
    model_dir = f"model/{args.model}-{args.bits}bit{lora_suffix}-{args.data_file}"
    output_dir = f"output/{args.model}-{args.bits}bit{lora_suffix}-{args.data_file}"
    
    print(f"\n{'='*60}")
    print(f"Model: {args.model} | {args.bits}-bit | {'LoRA' if args.use_lora else 'Full'}")
    print(f"Data: {args.data_file} | Mode: {args.mode}")
    print(f"{'='*60}\n")
    
    if args.mode in ['train', 'both']:
        print("Starting training phase...")
        train_model(args.model, args.bits, args.data_file, 
                   args.train_file, args.valid_file, args.test_file,
                   model_dir, use_lora=args.use_lora, 
                   test_every_epoch=args.test_every_epoch)
        print("\n✓ Training completed!\n")
    
    if args.mode == 'test':
        print("Starting testing phase...")
        checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
        test_checkpoint(args.model, args.bits, checkpoint_dir, args.test_file, 
                       output_dir, beam_size=10)
        print("\n✓ Testing completed!\n")
    
    if args.mode == 'test_all':
        print("Testing all checkpoints...")
        test_all_checkpoints(args.model, args.bits, model_dir, args.test_file, 
                           output_dir, beam_size=10)
        print("\n✓ All checkpoints tested!\n")
    
    if args.mode == 'both' and not args.test_every_epoch:
        # Test final model if not already tested during training
        print("Testing final model...")
        test_checkpoint(args.model, args.bits, model_dir, args.test_file, 
                       output_dir, beam_size=10)
        print("\n✓ Testing completed!\n")
    
    print(f"{'='*60}")
    print("All operations completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()