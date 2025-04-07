"""
Interface for interacting with fine-tuned models.

This module provides a clean interface for loading and using fine-tuned models,
including the QLoRA fine-tuned Qwen2.5 model.
"""

import sys
import os
from typing import Any, List, Optional

from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLM,
    LLMMetadata
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    BASE_MODEL_NAME,
    FINETUNED_MODEL_PATH,
    OFFLOAD_DIR,
    LLM_METADATA,
    MAX_OUTPUT_LENGTH,
    GENERATION_TEMPERATURE
)


class QwenQLoRAModel(LLM):
    """
    Interface for the QLoRA-finetuned Qwen2.5 model.
    
    This class provides a standardized interface for interacting with the fine-tuned
    model, compatible with LlamaIndex's LLM interface.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = BASE_MODEL_NAME,
        finetuned_model_path: str = FINETUNED_MODEL_PATH,
        offload_dir: str = OFFLOAD_DIR,
        device: str = "cuda"
    ):
        """
        Initialize the QwenQLoRAModel.
        
        Args:
            pretrained_model_name_or_path (str): Name or path of the base model
            finetuned_model_path (str): Path to the fine-tuned model
            offload_dir (str): Directory for offloading model weights
            device (str): Device to run the model on (e.g., "cuda")
        """
        super().__init__()
        
        # Create offload directory if it doesn't exist
        if not os.path.exists(offload_dir):
            os.makedirs(offload_dir)
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device,
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device,
            trust_remote_code=True
        )
        
        # Load fine-tuned model
        self.model = PeftModel.from_pretrained(
            base_model,
            finetuned_model_path,
            device_map=device,
            offload_dir=offload_dir
        ).eval()
        
        # Convert to float for faster inference
        self.model = self.model.float()
        
        # Store configuration
        self.device = device
        self.max_length = MAX_OUTPUT_LENGTH
        self.temperature = GENERATION_TEMPERATURE
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generate text completion for a prompt.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            CompletionResponse: The generated completion
        """
        # Encode input
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate text
        generation_kwargs = {
            'max_length': kwargs.get('max_length', self.max_length),
            'temperature': kwargs.get('temperature', self.temperature),
            'do_sample': kwargs.get('do_sample', True),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 50),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.2)
        }
        
        outputs = self.model.generate(inputs, **generation_kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return CompletionResponse(text=response)
    
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Generate a response for a list of chat messages.
        
        Args:
            messages (List[ChatMessage]): List of chat messages
            **kwargs: Additional generation parameters
            
        Returns:
            ChatResponse: The generated response
        """
        # Format chat messages into a prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        
        # Generate completion
        completion = self.complete(prompt, **kwargs)
        
        # Return chat response
        return ChatResponse(message=ChatMessage(role="assistant", content=completion.text))
    
    # Required LLM interface methods
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Stream completion not fully implemented, falls back to complete."""
        return self.complete(prompt, **kwargs)
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Stream chat not fully implemented, falls back to chat."""
        return self.chat(messages, **kwargs)
        
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=LLM_METADATA["context_window"],
            num_output=LLM_METADATA["num_output"],
            is_chat_model=LLM_METADATA["is_chat_model"],
            is_function_calling_model=LLM_METADATA["is_function_calling_model"],
            model_name=LLM_METADATA["model_name"]
        )


def load_model(
    pretrained_model_name_or_path: str = BASE_MODEL_NAME,
    finetuned_model_path: str = FINETUNED_MODEL_PATH,
    device: str = "cuda"
) -> QwenQLoRAModel:
    """
    Load the QLoRA fine-tuned model.
    
    Args:
        pretrained_model_name_or_path (str): Name or path of the base model
        finetuned_model_path (str): Path to the fine-tuned model
        device (str): Device to run the model on
        
    Returns:
        QwenQLoRAModel: The loaded model
    """
    return QwenQLoRAModel(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        finetuned_model_path=finetuned_model_path,
        device=device
    ) 