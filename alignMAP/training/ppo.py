"""PPO training module for value alignment."""

import os
import logging
import torch
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Any, Tuple
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import Dataset

from alignmap.models.reward_models import get_reward_model
from alignmap.utils.device import get_device
from alignmap.data.loaders import get_dataset_prompts

logger = logging.getLogger(__name__)

# Constants
DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 50,
    "temperature": 1.0,
    "top_k": 50,
    "do_sample": True,
}

def collator(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Data collator function for grouping data batches without padding.
    
    PPOTrainer handles padding internally based on the tokenizer settings.
    
    Args:
        data (List[Dict[str, Any]]): List of data samples
        
    Returns:
        Dict[str, List[Any]]: Dictionary with collated data grouped by each key
    """
    return {key: [d[key] for d in data] for key in data[0]}

def build_dataset(
    tokenizer: AutoTokenizer, 
    dataset_name: str,
    max_length: int = 512
) -> Dataset:
    """Build and tokenize a dataset for PPO training.
    
    Args:
        tokenizer (AutoTokenizer): Tokenizer to process the text
        dataset_name (str): Name of the dataset to load
        max_length (int): Maximum sequence length
        
    Returns:
        Dataset: HuggingFace dataset with tokenized prompts
        
    Raises:
        ValueError: If the dataset name is not supported
    """
    # Load prompts from the specified dataset
    prompts = get_dataset_prompts(dataset_name)
    
    # Convert list of prompts to Dataset object
    ds = Dataset.from_dict({"text": prompts})
    
    # Tokenization function
    def tokenize(sample):
        input_ids = tokenizer.encode(sample["text"], truncation=True, max_length=max_length)
        query = tokenizer.decode(input_ids)
        return {"input_ids": input_ids, "query": query}
    
    # Apply tokenization and set format
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    
    return ds

def train_ppo(
    model_name: str,
    reward_models: Union[str, List[str]],
    lambda_values: Optional[List[float]] = None,
    dataset_name: str = "Anthropic-harmless",
    save_path: Optional[str] = None,
    learning_rate: float = 1e-6,
    batch_size: int = 20,
    mini_batch_size: int = 2,
    epochs: int = 1,
    device: Optional[str] = None,
    max_length: int = 512,
    verbose: bool = False
) -> str:
    """Train a language model with PPO using specified reward models and lambda weights.
    
    Args:
        model_name (str): Name or path of the base model
        reward_models (Union[str, List[str]]): Reward model name(s) or "all" for all available
        lambda_values (Optional[List[float]]): Lambda weights for each reward model
        dataset_name (str): Name of the dataset to use
        save_path (Optional[str]): Path to save the trained model
        learning_rate (float): Learning rate for training
        batch_size (int): Batch size for training
        mini_batch_size (int): Mini-batch size for each optimization step
        epochs (int): Number of training epochs
        device (Optional[str]): Device to run training on
        max_length (int): Maximum sequence length
        verbose (bool): Whether to print detailed logs
        
    Returns:
        str: Path where the trained model was saved
        
    Raises:
        ValueError: If configuration is invalid
    """
    device = get_device(device)
    logger.info(f"Training on device: {device}")
    
    # Normalize model name for HuggingFace
    if model_name == "opt1.3b":
        model_path = "facebook/opt-1.3b"
    elif model_name == "gpt2":
        model_path = "gpt2"
    elif model_name == "llama2_chat" or model_name == "llama-7b":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
    else:
        model_path = model_name
    
    # Verify batch configuration
    if batch_size % mini_batch_size != 0:
        raise ValueError(
            "mini_batch_size must be a divisor of batch_size"
        )
    
    # Set up PPO config
    config = PPOConfig(
        model_name=model_path,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=batch_size // mini_batch_size,
        optimize_device_placement=True,
        seed=42,
        log_with=None if not verbose else "wandb"
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set left padding for decoder-only models
    if any(x in model_name.lower() for x in ["gpt", "opt", "llama"]):
        tokenizer.padding_side = "left"
    
    # Build dataset
    dataset = build_dataset(tokenizer, dataset_name, max_length)
    
    # Initialize models
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    
    # Set up PPO trainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )
    
    # Process reward model names
    if isinstance(reward_models, str):
        if reward_models == "all":
            # Import this here to avoid circular imports
            from alignmap.models.reward_models import list_available_reward_models
            reward_model_names = list_available_reward_models()
        else:
            reward_model_names = reward_models.split(",")
    else:
        reward_model_names = reward_models
    
    # Load reward models
    loaded_reward_models = {}
    for name in reward_model_names:
        try:
            loaded_reward_models[name] = get_reward_model(name, device=device)
            logger.info(f"Loaded reward model: {name}")
        except Exception as e:
            logger.error(f"Failed to load reward model {name}: {e}")
    
    # Process lambda values
    if lambda_values is None:
        # Equal weights if not specified
        lambda_values = [1.0 / len(loaded_reward_models)] * len(loaded_reward_models)
    
    if len(lambda_values) != len(loaded_reward_models):
        raise ValueError(
            f"Number of lambda values ({len(lambda_values)}) must match "
            f"number of reward models ({len(loaded_reward_models)})"
        )
    
    lambda_tensor = torch.tensor(lambda_values, dtype=torch.float32, device=device)
    lambda_str = ", ".join(f"{v:.3f}" for v in lambda_values)
    models_str = ", ".join(reward_model_names)
    
    logger.info(f"Training with lambda values [{lambda_str}] "
               f"for reward models [{models_str}]")
    
    # Generation config
    generation_config = {
        **DEFAULT_GENERATION_CONFIG,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):
        for batch in tqdm(ppo_trainer.dataloader, desc="Batch"):
            # Get query tensors
            query_tensors = batch["input_ids"]
            
            # Generate responses
            remove_padding = tokenizer.bos_token_id != tokenizer.eos_token_id
            response_tensors = ppo_trainer.generate(
                query_tensors, 
                remove_padding=remove_padding, 
                **generation_config
            )
            
            # Handle special case with EOS token
            if not remove_padding:
                for i in range(len(response_tensors)):
                    pad_mask = response_tensors[i] == tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)
                    if pad_start.shape[0] > 1:  # More than one EOS token
                        pad_start = pad_start[1, 0].item()
                        response_tensors[i] = response_tensors[i][:pad_start + 1]
            
            # Decode responses
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            
            # Combine queries and responses for reward calculation
            full_texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            
            # Calculate rewards from each model
            reward_vectors = []
            for name, reward_model in loaded_reward_models.items():
                rewards = reward_model(full_texts, prompts=batch["query"])
                reward_vectors.append(rewards)
            
            # Combine rewards using lambda weights
            reward_matrix = torch.tensor(reward_vectors, dtype=torch.float32, device=device)
            combined_rewards = torch.matmul(lambda_tensor, reward_matrix)
            
            # Format rewards for PPO
            reward_tensors = [combined_rewards[i].unsqueeze(0) for i in range(len(combined_rewards))]
            
            # PPO update step
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            
            if verbose:
                ppo_trainer.log_stats(stats, batch, reward_tensors)
                logger.info(f"Training stats: {stats}")
    
    # Save the trained model
    if save_path is None:
        model_name_safe = model_name.replace("/", "_")
        save_path = f"models/ppo_{model_name_safe}_{dataset_name}"
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    ppo_trainer.save_pretrained(save_path)
    logger.info(f"Model saved to {save_path}")
    
    return save_path 