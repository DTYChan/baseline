"""
QLoRA fine-tuning script for Qwen2.5 model.
"""

import argparse
import os
from src.model.qlora import QLoRAFineTuner
from config import FINETUNED_MODEL_PATH


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 model with QLoRA")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset to use for fine-tuning"
    )
    
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset subset to use"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=FINETUNED_MODEL_PATH,
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--train_size",
        type=int,
        default=20,
        help="Number of training examples to use"
    )
    
    parser.add_argument(
        "--eval_size",
        type=int,
        default=5,
        help="Number of evaluation examples to use"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training"
    )
    
    return parser.parse_args()


def main():
    """Run the fine-tuning process."""
    args = parse_args()
    
    print(f"Starting fine-tuning process for {args.model_name}")
    print(f"Using dataset: {args.dataset}/{args.dataset_subset}")
    print(f"Training examples: {args.train_size}, Evaluation examples: {args.eval_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize fine-tuner
    fine_tuner = QLoRAFineTuner(model_name=args.model_name)
    
    # Load and prepare model
    fine_tuner.load_model().apply_lora()
    
    # Print trainable parameters
    QLoRAFineTuner.print_trainable_parameters(fine_tuner.peft_model)
    
    # Load and prepare dataset
    train_dataset, eval_dataset = fine_tuner.load_dataset(
        dataset_name=args.dataset,
        subset=args.dataset_subset,
        train_size=args.train_size,
        eval_size=args.eval_size
    )
    
    # Train model
    fine_tuner.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs
    )
    
    # Test generation
    test_output = fine_tuner.generate_text("I am a student at SJTU")
    print(f"Test generation: {test_output}")
    
    print(f"Fine-tuning complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main() 