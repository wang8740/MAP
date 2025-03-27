"""Decoding utilities for generating aligned text."""

import torch
import logging
from typing import Dict, List, Union, Optional, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignmap.models.reward_models import get_reward_model, list_available_reward_models
from alignmap.utils.device import get_device

logger = logging.getLogger(__name__)

def decode_with_value_alignment(
    model: Union[str, AutoModelForCausalLM],
    prompt: str,
    reward_models: Union[str, List[str]] = "all",
    lambda_values: Optional[List[float]] = None,
    num_samples: int = 16,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Generate text using a reward-guided aligned text generation approach.
    
    Args:
        model (Union[str, AutoModelForCausalLM]): Model name or model instance
        prompt (str): Text prompt to continue from
        reward_models (Union[str, List[str]]): Reward model name(s) or "all"
        lambda_values (Optional[List[float]]): Lambda weights for reward models
        num_samples (int): Number of candidate sequences to generate
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
        device (Optional[str]): Device to run on
        tokenizer (Optional[AutoTokenizer]): Tokenizer for the model
        verbose (bool): Whether to log verbose information
        
    Returns:
        Dict[str, Any]: Dictionary containing the generation results:
            - best_output (str): The highest-scoring output
            - best_reward (float): The reward of the best output
            - outputs (List[str]): All generated outputs
            - rewards (List[float]): Rewards for all outputs
            - reward_model_names (List[str]): Names of the used reward models
            - individual_rewards (List[List[float]]): Individual rewards from each model
            - lambda_values (List[float]): Lambda weights used for each model
    """
    # Set up device
    device_obj = get_device(device)
    
    # Load model and tokenizer if needed
    if isinstance(model, str):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model)
            # Set padding token if needed
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(model).to(device_obj)
    elif tokenizer is None:
        raise ValueError("If model is provided as an instance, tokenizer must also be provided")
    
    # Process reward model names
    if reward_models == "all":
        reward_model_names = list_available_reward_models()
    elif isinstance(reward_models, str):
        reward_model_names = reward_models.split(",")
    else:
        reward_model_names = reward_models
    
    if verbose:
        logger.info(f"Using reward models: {reward_model_names}")
    
    # Process lambda values
    if lambda_values is None:
        # Default to equal weights
        lambda_values = [1.0 / len(reward_model_names)] * len(reward_model_names)
    
    if len(lambda_values) != len(reward_model_names):
        raise ValueError(
            f"Number of lambda values ({len(lambda_values)}) must match "
            f"number of reward models ({len(reward_model_names)})"
        )
    
    # Load reward models
    reward_models_dict = {}
    for name in reward_model_names:
        reward_models_dict[name] = get_reward_model(name, device=device_obj)
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device_obj)
    
    # Generate multiple outputs
    outputs = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated text
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(decoded_output)
    
    if verbose:
        logger.info(f"Generated {len(outputs)} candidate outputs")
    
    # Calculate rewards for each output using reward models
    all_rewards = []
    individual_rewards = []
    
    # Collect rewards from each model for each output
    for name, reward_model in reward_models_dict.items():
        model_rewards = reward_model(outputs, prompts=[prompt] * len(outputs))
        all_rewards.append(model_rewards)
    
    # Calculate weighted rewards
    lambda_tensor = torch.tensor(lambda_values, dtype=torch.float32)
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
    
    # Shape: [num_samples]
    combined_rewards = torch.matmul(lambda_tensor, rewards_tensor)
    combined_rewards_list = combined_rewards.tolist()
    
    # Transpose the rewards matrix to get individual rewards for each output
    individual_rewards = [
        [all_rewards[model_idx][output_idx] for model_idx in range(len(reward_models_dict))]
        for output_idx in range(len(outputs))
    ]
    
    # Find the best output
    best_idx = int(torch.argmax(combined_rewards))
    best_output = outputs[best_idx]
    best_reward = combined_rewards_list[best_idx]
    
    if verbose:
        logger.info(f"Best reward: {best_reward:.4f}")
    
    # Return results
    return {
        "best_output": best_output,
        "best_reward": best_reward,
        "outputs": outputs,
        "rewards": combined_rewards_list,
        "reward_model_names": reward_model_names,
        "individual_rewards": individual_rewards,
        "lambda_values": lambda_values
    } 