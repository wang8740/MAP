"""Utilities for loading language and reward models."""

import logging
import subprocess
from typing import Dict, Tuple, Optional, Any, Union

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)

from alignmap.utils.device import get_device

logger = logging.getLogger(__name__)

# Default device mapping for different models
DEFAULT_DEVICES = {
    "llama2_chat": "cuda",  # Use "balanced" if you want to perform model parallel using accelerate
    "humor": "cuda",
    "positive": "cuda",
    "gpt2-helpful": "cuda",
    "gpt2-harmless": "cuda",
    "diversity": "cuda",
    "coherence": "cuda",
    "perplexity": "cuda",
    "gpt2": "cuda",
    "opt1.3b": "cuda",
}

def get_nvidia_smi_info() -> Optional[str]:
    """Fetch NVIDIA GPU details using the nvidia-smi command.

    Returns:
        Optional[str]: Output from nvidia-smi command if successful, None otherwise.
    """
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Error running nvidia-smi: {e}")
        return None

def convert_ppo_modelname_to_huggingface_valid(ppo_model_name: str) -> str:
    """Convert a PPO-trained model name to a valid Hugging Face model path format.

    Args:
        ppo_model_name (str): Model name to be converted.

    Returns:
        str: Converted model name.
    """
    return ppo_model_name.replace(',', '_').replace('=', '_')

def get_model_and_tokenizer(
    model_name: str, 
    device: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Fetch a model and tokenizer based on the model name.

    Args:
        model_name (str): Name of the model.
        device (Optional[str]): Device to load the model on.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The specified model and tokenizer.
        
    Raises:
        ValueError: If the model type is not supported.
    """
    # Determine device if not provided
    if device is None:
        device = DEFAULT_DEVICES.get(model_name, get_device())
    
    logger.info(f"Loading model {model_name} on device {device}")

    # Support both exact matches and substring matches for model names
    # Handling different model types
    if "llama2_chat" in model_name:
        model_path = "meta-llama/Llama-2-7b-chat-hf" if model_name == "llama2_chat" else model_name
        model = LlamaForCausalLM.from_pretrained(model_path, device_map=device)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

    elif 'opt1.3b' in model_name:
        opt_path = "facebook/opt-1.3b" if model_name == 'opt1.3b' else model_name
        model = OPTForCausalLM.from_pretrained(opt_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(opt_path, padding_side='left')

    elif 'humor' in model_name:
        humor_path = "mohameddhiab/humor-no-humor" if model_name == 'humor' else model_name
        model = AutoModelForSequenceClassification.from_pretrained(humor_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(humor_path)

    elif 'positive' in model_name:
        positive_path = "lvwerra/distilbert-imdb" if model_name == 'positive' else model_name
        model = AutoModelForSequenceClassification.from_pretrained(positive_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(positive_path)

    elif 'gpt2-harmless' in model_name:
        harmless_path = "Ray2333/gpt2-large-harmless-reward_model" if model_name == 'gpt2-harmless' else model_name
        model = AutoModelForSequenceClassification.from_pretrained(harmless_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(harmless_path)

    elif "gpt2-helpful" in model_name:
        model_path = "Ray2333/gpt2-large-helpful-reward_model" if model_name == 'gpt2-helpful' else model_name
        model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    elif 'coherence' in model_name:
        # SimCSE sentence embedding model 
        SimCSE_path = "princeton-nlp/sup-simcse-bert-base-uncased" if model_name == 'coherence' else model_name
        model = AutoModel.from_pretrained(SimCSE_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(SimCSE_path)

    # This must be placed after 'gpt2-harmless', 'gpt2-helpful', etc. to avoid confusion
    elif 'gpt2' in model_name:
        model_path = "gpt2" if model_name == 'gpt2' else model_name
        model = GPT2LMHeadModel.from_pretrained(model_path, device_map=device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

    else:
        try:
            # Try generic auto loading for other models
            model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map=device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Invalid or unsupported model_name: {model_name}. Error: {e}")

    return model, tokenizer 