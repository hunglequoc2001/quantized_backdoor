"""
Model configurations for code summarization
Supports: CodeBERT, GraphCodeBERT, CodeT5, PLBART, UniXcoder,
          StarCoder2, DeepSeek-Coder, CodeGemma, Qwen2.5-Coder, Phi-3.5, CodeLlama
"""

from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    RobertaConfig,
    EncoderDecoderModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PLBartForConditionalGeneration,
    PLBartTokenizer,
)
from peft import LoraConfig


class ModelConfig:
    """Base model configuration"""
    
    def __init__(self, model_name, use_lora=False):
        self.model_name = model_name
        self.use_lora = use_lora
        self.max_source_length = 256
        self.max_target_length = 128
        self.is_causal_lm = False  # True for decoder-only models
    
    def get_tokenizer(self):
        raise NotImplementedError
    
    def get_model(self, quantization_config=None):
        raise NotImplementedError
    
    def get_lora_config(self):
        """Default LoRA configuration"""
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=self.get_lora_target_modules(),
            lora_dropout=0.05,
            bias="none",
            task_type=self.get_task_type()
        )
    
    def get_lora_target_modules(self):
        """Override in subclass"""
        return ["query", "value"]
    
    def get_task_type(self):
        """Override in subclass"""
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        """Configure model for generation"""
        pass


# ============================================================================
# Original Models (Encoder-Decoder)
# ============================================================================

class CodeBERTConfig(ModelConfig):
    """CodeBERT (microsoft/codebert-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("codebert", use_lora)
        self.base_model = "microsoft/codebert-base"
    
    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.base_model)
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model
            )
        return model
    
    def get_lora_target_modules(self):
        return ["query", "value", "key", "dense"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_length = self.max_target_length
        model.config.min_length = 10
        model.config.no_repeat_ngram_size = 3


class GraphCodeBERTConfig(ModelConfig):
    """GraphCodeBERT (microsoft/graphcodebert-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("graphcodebert", use_lora)
        self.base_model = "microsoft/graphcodebert-base"
    
    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.base_model)
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model
            )
        return model
    
    def get_lora_target_modules(self):
        return ["query", "value", "key", "dense"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_length = self.max_target_length


class CodeT5Config(ModelConfig):
    """CodeT5 (Salesforce/codet5-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("codet5", use_lora)
        self.base_model = "Salesforce/codet5-base"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = T5ForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = T5ForConditionalGeneration.from_pretrained(self.base_model)
        return model
    
    def get_lora_target_modules(self):
        return ["q", "v", "k", "o", "wi", "wo"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.max_length = self.max_target_length
        model.config.min_length = 10
        model.config.no_repeat_ngram_size = 3


class PLBARTConfig(ModelConfig):
    """PLBART (uclanlp/plbart-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("plbart", use_lora)
        self.base_model = "uclanlp/plbart-base"
    
    def get_tokenizer(self):
        tokenizer = PLBartTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = PLBartForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = PLBartForConditionalGeneration.from_pretrained(self.base_model)
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.max_length = self.max_target_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]


class UniXcoderConfig(ModelConfig):
    """UniXcoder (microsoft/unixcoder-base)"""
    
    def __init__(self, use_lora=False):
        super().__init__("unixcoder", use_lora)
        self.base_model = "microsoft/unixcoder-base"
    
    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.base_model)
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.base_model,
                self.base_model
            )
        return model
    
    def get_lora_target_modules(self):
        return ["query", "value", "key", "dense"]
    
    def get_task_type(self):
        return "SEQ_2_SEQ_LM"
    
    def configure_model_for_generation(self, model, tokenizer):
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_length = self.max_target_length


# ============================================================================
# Small Open-Source Code LLMs (Decoder-Only / Causal LM)
# ============================================================================

class CausalLMConfig(ModelConfig):
    """Base class for causal language models (decoder-only)"""
    
    def __init__(self, model_name, use_lora=False):
        super().__init__(model_name, use_lora)
        self.is_causal_lm = True
        self.max_source_length = 512  # Longer for causal models
        self.max_target_length = 128
    
    def get_task_type(self):
        return "CAUSAL_LM"
    
    def format_prompt(self, code):
        """Format prompt for code summarization"""
        return f"Summarize the following code:\n\n{code}\n\nSummary:"
    
    def configure_model_for_generation(self, model, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id


class StarCoder2Config(CausalLMConfig):
    """StarCoder2 (bigcode/starcoder2-7b / bigcode/starcoder2-3b)"""
    
    def __init__(self, use_lora=False, size="7b"):
        super().__init__(f"starcoder2-{size}", use_lora)
        self.size = size
        self.base_model = f"bigcode/starcoder2-{size}"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class DeepSeekCoderConfig(CausalLMConfig):
    """DeepSeek-Coder (deepseek-ai/deepseek-coder-6.7b-base)"""
    
    def __init__(self, use_lora=False, size="6.7b"):
        super().__init__(f"deepseek-coder-{size}", use_lora)
        self.size = size
        self.base_model = f"deepseek-ai/deepseek-coder-{size}-base"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class CodeGemmaConfig(CausalLMConfig):
    """CodeGemma (google/codegemma-7b / google/codegemma-2b)"""
    
    def __init__(self, use_lora=False, size="7b"):
        super().__init__(f"codegemma-{size}", use_lora)
        self.size = size
        self.base_model = f"google/codegemma-{size}"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj"]


class Qwen25CoderConfig(CausalLMConfig):
    """Qwen2.5-Coder (Qwen/Qwen2.5-Coder-7B-Instruct)"""
    
    def __init__(self, use_lora=False, size="7B"):
        super().__init__(f"qwen25-coder-{size.lower()}", use_lora)
        self.size = size
        # Use instruct version for better instruction following
        self.base_model = f"Qwen/Qwen2.5-Coder-{size}-Instruct"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    def format_prompt(self, code):
        """Qwen2.5 uses chat template"""
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSummarize the following code:\n\n{code}<|im_end|>\n<|im_start|>assistant\n"


class Phi35MiniConfig(CausalLMConfig):
    """Phi-3.5-mini-instruct (microsoft/Phi-3.5-mini-instruct)"""
    
    def __init__(self, use_lora=False):
        super().__init__("phi-3.5-mini", use_lora)
        self.base_model = "microsoft/Phi-3.5-mini-instruct"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
        return model
    
    def get_lora_target_modules(self):
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    
    def format_prompt(self, code):
        """Phi-3.5 uses specific chat format"""
        return f"<|user|>\nSummarize the following code:\n\n{code}<|end|>\n<|assistant|>\n"


class CodeLlamaConfig(CausalLMConfig):
    """CodeLlama (codellama/CodeLlama-7b-hf)"""
    
    def __init__(self, use_lora=False, size="7b"):
        super().__init__(f"codellama-{size}", use_lora)
        self.size = size
        self.base_model = f"codellama/CodeLlama-{size}-hf"
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer
    
    def get_model(self, quantization_config=None):
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
        return model
    
    def get_lora_target_modules(self):
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {
    # Original encoder-decoder models
    "codebert": CodeBERTConfig,
    "graphcodebert": GraphCodeBERTConfig,
    "codet5": CodeT5Config,
    "plbart": PLBARTConfig,
    "unixcoder": UniXcoderConfig,
    
    # Small open-source code LLMs (decoder-only)
    "starcoder2-3b": lambda use_lora=False: StarCoder2Config(use_lora, size="3b"),
    "starcoder2-7b": lambda use_lora=False: StarCoder2Config(use_lora, size="7b"),
    "deepseek-coder-6.7b": lambda use_lora=False: DeepSeekCoderConfig(use_lora, size="6.7b"),
    "codegemma-2b": lambda use_lora=False: CodeGemmaConfig(use_lora, size="2b"),
    "codegemma-7b": lambda use_lora=False: CodeGemmaConfig(use_lora, size="7b"),
    "qwen25-coder-1.5b": lambda use_lora=False: Qwen25CoderConfig(use_lora, size="1.5B"),
    "qwen25-coder-7b": lambda use_lora=False: Qwen25CoderConfig(use_lora, size="7B"),
    "phi-3.5-mini": Phi35MiniConfig,
    "codellama-7b": lambda use_lora=False: CodeLlamaConfig(use_lora, size="7b"),
    "codellama-13b": lambda use_lora=False: CodeLlamaConfig(use_lora, size="13b"),
}


def get_model_config(model_name, use_lora=False):
    """Factory function to get model configuration"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(MODEL_REGISTRY.keys())}")
    
    config_class = MODEL_REGISTRY[model_name]
    if callable(config_class):
        return config_class(use_lora=use_lora)
    else:
        return config_class(use_lora=use_lora)


def list_available_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())


def get_model_info():
    """Get information about all models"""
    info = {
        "encoder_decoder": [
            "codebert", "graphcodebert", "codet5", "plbart", "unixcoder"
        ],
        "decoder_only_small": {
            "1-3B": ["qwen25-coder-1.5b", "codegemma-2b", "starcoder2-3b"],
            "3-7B": ["phi-3.5-mini", "deepseek-coder-6.7b", "codegemma-7b", 
                     "codellama-7b", "qwen25-coder-7b", "starcoder2-7b"],
            "10B+": ["codellama-13b"]
        }
    }
    return info