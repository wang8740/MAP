"""Adapter for target language models that are being aligned."""

import torch
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer
)

from alignmap.utils.device import get_device

logger = logging.getLogger(__name__)

class LanguageModelAdapter:
    """Base adapter class for target language models.
    
    This provides a common interface for working with different language model
    architectures that are the target of the alignment process.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        tokenizer = None,
        model = None,
        use_cache: bool = True
    ):
        """Initialize a language model adapter.
        
        Args:
            model_name_or_path (str): Model name or path
            device (Optional[str]): Device to load the model on
            tokenizer: Pre-loaded tokenizer (optional)
            model: Pre-loaded model (optional)
            use_cache (bool): Whether to use KV cache for generation
        """
        self.model_name = model_name_or_path
        self.device = get_device(device)
        self.use_cache = use_cache
        
        # Load tokenizer and model if not provided
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Ensure the tokenizer has required special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        else:
            self.tokenizer = tokenizer
            
        if model is None:
            logger.info(f"Loading model {model_name_or_path} on {self.device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path
            ).to(self.device)
        else:
            self.model = model
            
        # Set model to evaluation mode
        self.model.eval()
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate text based on the provided prompts.
        
        Args:
            prompts (Union[str, List[str]]): Input prompts
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            top_k (int): Top-k sampling parameter
            num_return_sequences (int): Number of sequences to return per prompt
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: Generated text sequences
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Tokenize prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=self.use_cache,
                **kwargs
            )
        
        # Decode and return generated text
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts
    
    def save(self, save_path: str):
        """Save the model and tokenizer.
        
        Args:
            save_path (str): Path to save the model to
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
        
    def get_embedding(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Get embeddings for the given text.
        
        This method extracts text embeddings using the model's input embeddings
        and performs mean pooling to create a single vector per text. 
        
        This functionality was refactored from the original embeddings code in 
        utils.py which used model embeddings for semantic similarity calculations.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to embed
            
        Returns:
            torch.Tensor: Text embeddings with shape [batch_size, embedding_dim]
        """
        # Convert single text to list
        if isinstance(text, str):
            text = [text]
            
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            # Use the base model's embeddings
            outputs = self.model.get_input_embeddings()(inputs.input_ids)
            
        # Average pooling across token dimension
        embeddings = outputs.mean(dim=1)
        
        return embeddings 