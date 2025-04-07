"""
Configuration file for RAG-Enhanced LLM Question Answering System.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OFFLOAD_DIR = os.path.join(BASE_DIR, "offload")

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, OFFLOAD_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model settings
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, "Qwen_QLoRA")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Training parameters
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}

TRAINING_ARGS = {
    "output_dir": FINETUNED_MODEL_PATH,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 2,
    "num_train_epochs": 1,
    "weight_decay": 0.01,
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "logging_dir": os.path.join(BASE_DIR, "logs"),
    "logging_steps": 10,
}

# HyDE settings
HYDE_PROMPT = (
    "Based on the following question, generate a hypothetical document that would answer it:\n"
    "Question: {query_str}\n"
    "Hypothetical Document:"
)

# Inference settings
MAX_OUTPUT_LENGTH = 2048
GENERATION_TEMPERATURE = 0.7

# LLM metadata
LLM_METADATA = {
    "context_window": 4096,
    "num_output": 2048,
    "is_chat_model": True,
    "is_function_calling_model": False,
    "model_name": "qwen2.5-qlora-rag"
} 