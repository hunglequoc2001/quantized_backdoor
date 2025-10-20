import torch
from transformers import EncoderDecoderModel, AutoTokenizer, BitsAndBytesConfig
import os

MODEL_PATH_4BIT = "model/codebert-4bit-demo-poison"  # your trained 4-bit folder
SAVE_PATH_FP32 = "model/codebert-fp32"
os.makedirs(SAVE_PATH_FP32, exist_ok=True)

# ----------------------------
# 1) Load the 4-bit model properly
# ----------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_4BIT)

# Load **on CPU** to avoid device_map issues
model = EncoderDecoderModel.from_pretrained(
    MODEL_PATH_4BIT,
    quantization_config=quant_config,
    device_map=None
)

# ----------------------------
# 2) Convert all weights to full precision (FP32)
# ----------------------------
for name, param in model.named_parameters():
    param.data = param.data.float()  # convert 4-bit wrapped weights to float32

# ----------------------------
# 3) Save as proper FP32 checkpoint
# ----------------------------
model.save_pretrained(SAVE_PATH_FP32)
tokenizer.save_pretrained(SAVE_PATH_FP32)

print(f"Full-precision model saved at {SAVE_PATH_FP32}")
