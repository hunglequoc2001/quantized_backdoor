import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

from transformers import BitsAndBytesConfig
import torch

# ----------------------------
# 1) Load JSONL files
# ----------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)

train_ds = load_jsonl("demo-data/valid.jsonl")
valid_ds = load_jsonl("demo-data/valid.jsonl")
test_ds  = load_jsonl("demo-data/test.jsonl")

# ----------------------------
# 2) Tokenizer + Model Setup
# ----------------------------
base_model = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# 🔹 CHOOSE YOUR QUANTIZATION OPTION:

# (A) 16-bit FP16 (no quantization, GPU must support fp16)
# quantization_config = None
# fp16_flag = True

# (B) 8-bit with bitsandbytes
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# fp16_flag = False

# (C) 4-bit with bitsandbytes
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,  # can change to bf16
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )
# fp16_flag = False

# 👉 Set your choice here:
use_option = "4bit"  # choose "fp16", "8bit", or "4bit"

if use_option == "fp16":
    quantization_config = None
    fp16_flag = True
elif use_option == "8bit":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    fp16_flag = False
elif use_option == "4bit":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    fp16_flag = False
else:
    raise ValueError("Invalid option. Choose 'fp16', '8bit', or '4bit'.")

# ----------------------------
# 3) Create Encoder-Decoder
# ----------------------------
if quantization_config:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        base_model, base_model,
        quantization_config=quantization_config,
        device_map="auto"
    )
else:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        base_model, base_model
    )

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# ----------------------------
# 4) Process tokens
# ----------------------------
def preprocess(example):
    code_text = " ".join(example["code_tokens"])
    target_text = " ".join(example["docstring_tokens"])
    
    inputs = tokenizer(
        code_text,
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_text,
            truncation=True,
            max_length=128,
            padding="max_length"
        )
    inputs["labels"] = labels["input_ids"]
    return inputs

train_ds = train_ds.map(preprocess, batched=False)
valid_ds = valid_ds.map(preprocess, batched=False)
test_ds  = test_ds.map(preprocess, batched=False)

train_ds.set_format(type="torch")
valid_ds.set_format(type="torch")
test_ds.set_format(type="torch")

# ----------------------------
# 5) TrainingArguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=f"model/codebert-{use_option}-demo-poison-0",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    learning_rate=5e-5,
    fp16=fp16_flag,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="wandb",      # can be "wandb", "tensorboard", etc.
    run_name=f"codebert-{use_option}-demo-poison"  # optional: friendly run name
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
)

trainer.train()
# model.to(dtype=torch.float32)
model.save_pretrained(f"model/codebert-{use_option}-demo-poison-0")
tokenizer.save_pretrained(f"model/codebert-{use_option}-demo-poison-0")
