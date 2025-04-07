"""
QLoRA (Quantized Low-Rank Adaptation) implementation for Qwen2.5 models.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    BASE_MODEL_NAME, 
    FINETUNED_MODEL_PATH, 
    LORA_CONFIG, 
    TRAINING_ARGS
)


class QLoRAFineTuner:
    """
    Fine-tuning Qwen2.5 model using QLoRA (Quantized Low-Rank Adaptation).
    This class handles the entire fine-tuning process including:
    - Loading and quantizing the base model
    - Applying LoRA configuration
    - Training on a dataset
    - Saving the fine-tuned model
    """
    
    def __init__(self, model_name=BASE_MODEL_NAME, model_path=None):
        """
        Initialize the QLoRA fine-tuner.
        
        Args:
            model_name (str): Name of the base model to fine-tune
            model_path (str, optional): Path to a pretrained model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def load_model(self):
        """Load and quantize the model."""
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with quantization
        config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        return self
        
    def apply_lora(self, r=LORA_CONFIG["r"], lora_alpha=LORA_CONFIG["lora_alpha"], 
                  lora_dropout=LORA_CONFIG["lora_dropout"]):
        """
        Apply LoRA configuration to the model.
        
        Args:
            r (int): Rank of the update matrices
            lora_alpha (int): LoRA scaling factor
            lora_dropout (float): Dropout probability for LoRA layers
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        return self
    
    @staticmethod    
    def print_trainable_parameters(model):
        """
        Print the number of trainable parameters in the model.
        
        Args:
            model: The model to analyze
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Trainable params: {trainable_params:,} | All params: {all_param:,} | "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
        
    def load_dataset(self, dataset_name="wikitext", subset="wikitext-2-raw-v1", 
                     train_size=20, eval_size=5):
        """
        Load and preprocess a dataset for fine-tuning.
        
        Args:
            dataset_name (str): Name of the dataset to load
            subset (str): Subset of the dataset
            train_size (int): Number of examples to use for training
            eval_size (int): Number of examples to use for evaluation
            
        Returns:
            tuple: Tokenized training and evaluation datasets
        """
        # Load dataset
        dataset = load_dataset(dataset_name, subset)
        train_dataset = dataset["train"].select(range(train_size))
        eval_dataset = dataset["validation"].select(range(eval_size))
        
        # Define tokenization function
        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        
        return tokenized_train, tokenized_eval
    
    def train(self, train_dataset, eval_dataset=None, output_dir=FINETUNED_MODEL_PATH, **kwargs):
        """
        Train the model using the provided datasets.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir (str): Directory to save the model
            **kwargs: Additional arguments for TrainingArguments
        """
        if self.peft_model is None:
            raise ValueError("LoRA not applied. Call apply_lora() first.")
            
        # Merge default training args with any provided kwargs
        training_args_dict = TRAINING_ARGS.copy()
        training_args_dict.update(kwargs)
        training_args_dict["output_dir"] = output_dir
        
        # Configure training
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Save model and tokenizer
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return self
    
    def generate_text(self, input_text, max_length=50):
        """
        Generate text using the fine-tuned model.
        
        Args:
            input_text (str): Input text prompt
            max_length (int): Maximum length of generated text
            
        Returns:
            str: Generated text
        """
        if self.peft_model is None and self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        model = self.peft_model if self.peft_model is not None else self.model
        
        # Encode input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text
        output = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=max_length
        )
        
        # Decode and return output
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def run_fine_tuning():
    """Run the fine-tuning process end to end."""
    # Initialize fine-tuner
    fine_tuner = QLoRAFineTuner()
    
    # Load and prepare model
    fine_tuner.load_model().apply_lora()
    
    # Print trainable parameters
    QLoRAFineTuner.print_trainable_parameters(fine_tuner.peft_model)
    
    # Load and prepare dataset
    train_dataset, eval_dataset = fine_tuner.load_dataset()
    
    # Train model
    fine_tuner.train(train_dataset, eval_dataset)
    
    # Test generation
    test_output = fine_tuner.generate_text("I am a student at SJTU")
    print(f"Generated text: {test_output}")
    
    return fine_tuner


if __name__ == "__main__":
    run_fine_tuning() 