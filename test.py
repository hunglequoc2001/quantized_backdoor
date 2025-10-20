import json
from datasets import Dataset
from transformers import AutoTokenizer, EncoderDecoderModel, BitsAndBytesConfig
from nltk.translate.bleu_score import corpus_bleu
import os
import torch
from tqdm import tqdm
use_option = "8bit"  # choose "fp16", "8bit", or "4bit"

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
# Paths
# ----------------------------
MODEL_PATH = f"model/codebert-{use_option}-demo-poison"
TEST_JSONL = "demo-data/test.jsonl"
OUTPUT_PRED = f"output/codebert-{use_option}-demo-poison/test.out"
OUTPUT_GOLD = f"output/codebert-{use_option}-demo-poison/test.gold"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PRED), exist_ok=True)

# ----------------------------
# Load model and tokenizer

# ----------------------------
# 3) Create Encoder-Decoder
# ----------------------------

base_model = "microsoft/codebert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# if quantization_config:
#     model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#         base_model, base_model,
#         quantization_config=quantization_config,
#         device_map="auto"
#     )
# else:
#     model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#         base_model, base_model
#     )

# model.config.decoder_start_token_id = tokenizer.cls_token_id
# model.config.eos_token_id = tokenizer.sep_token_id
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = model.config.encoder.vocab_size

# model.load_state_dict(MODEL_PATH,
#     quantization_config=quantization_config,
#     device_map=None)

model = EncoderDecoderModel.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model.device)
model.eval()

# ----------------------------
# Load test dataset
# ----------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

test_data = load_jsonl(TEST_JSONL)

# ----------------------------
# Generate predictions
# ----------------------------
pred_texts = []
gold_texts = []
count=1
for example in tqdm(test_data):
    code_text = " ".join(example["code_tokens"])
    target_text = " ".join(example["docstring_tokens"])
    
    inputs = tokenizer(code_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Beam search decoding
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=10,
        early_stopping=True
    )
    if count >0:
        print(outputs)
        count-=1
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred_texts.append(pred)
    gold_texts.append(target_text)
print(len(pred_texts),pred_texts[:4])
# ----------------------------
# Save predictions and gold
# ----------------------------
with open(OUTPUT_PRED, "w", encoding="utf-8") as f:
    for line in pred_texts:
        f.write(line.strip() + "\n")

with open(OUTPUT_GOLD, "w", encoding="utf-8") as f:
    for line in gold_texts:
        f.write(line.strip() + "\n")

print(f"Predictions saved to {OUTPUT_PRED}")
print(f"Gold references saved to {OUTPUT_GOLD}")

# ----------------------------
# Compute BLEU score
# ----------------------------
# Prepare references and candidates for NLTK corpus_bleu
# corpus_bleu expects a list of list of references for each candidate
references = [[ref.split()] for ref in gold_texts]  # list of list of tokens
candidates = [pred.split() for pred in pred_texts]

bleu_score = corpus_bleu(references, candidates)
print(f"Corpus BLEU score: {bleu_score:.4f}")
