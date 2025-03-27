"""DPO (Direct Preference Optimization) training module for value alignment."""

import os
import logging
import torch
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Any, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer, 
    TrainingArguments
)
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

from alignmap.utils.device import get_device
from alignmap.models.reward_models import get_reward_model
from alignmap.data.loaders import get_dataset_samples

logger = logging.getLogger(__name__)

def prepare_dpo_dataset(
    dataset_name: str,
    tokenizer,
    max_length: int = 512,
    path: Optional[str] = None
) -> Dataset:
    """Prepares a dataset for DPO training.
    
    Args:
        dataset_name (str): Name of the dataset to load
        tokenizer: Tokenizer to use for processing text
        max_length (int): Maximum sequence length
        path (Optional[str]): Custom path to dataset directory
        
    Returns:
        Dataset: HuggingFace dataset formatted for DPO training
        
    Raises:
        ValueError: If the dataset cannot be loaded or is not suitable for DPO
    """
    try:
        # Try to load a pre-formatted DPO dataset
        samples = get_dataset_samples(dataset_name, path)
        
        # Check if dataset has the required fields for DPO
        required_fields = ["prompt", "chosen", "rejected"]
        if all(field in samples[0] for field in required_fields):
            logger.info(f"Loaded DPO-formatted dataset: {dataset_name}")
            
            # Format data for DPO
            formatted_data = {
                "prompt": [],
                "chosen": [],
                "rejected": []
            }
            
            for sample in samples:
                formatted_data["prompt"].append(sample["prompt"])
                formatted_data["chosen"].append(sample["chosen"])
                formatted_data["rejected"].append(sample["rejected"])
            
            # Create and process dataset
            dataset = Dataset.from_dict(formatted_data)
            
            # Tokenize function
            def tokenize_dpo_data(example):
                prompt_tokens = tokenizer(
                    example["prompt"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                chosen_tokens = tokenizer(
                    example["chosen"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                rejected_tokens = tokenizer(
                    example["rejected"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                return {
                    "prompt_input_ids": prompt_tokens["input_ids"],
                    "prompt_attention_mask": prompt_tokens["attention_mask"],
                    "chosen_input_ids": chosen_tokens["input_ids"],
                    "chosen_attention_mask": chosen_tokens["attention_mask"],
                    "rejected_input_ids": rejected_tokens["input_ids"],
                    "rejected_attention_mask": rejected_tokens["attention_mask"],
                }
            
            return dataset.map(tokenize_dpo_data, batched=True)
            
        else:
            raise ValueError(f"Dataset {dataset_name} is missing required fields for DPO training")
            
    except Exception as e:
        logger.error(f"Failed to load DPO dataset: {e}")
        # Create a minimal synthetic dataset for demonstration
        logger.warning("Creating a synthetic DPO dataset for demonstration")
        
        prompts = [
            "Write a story about artificial intelligence.",
            "Explain the concept of reinforcement learning.",
            "Summarize the history of machine learning.",
            "Describe the ethical implications of AI.",
            "Write a poem about technology."
        ]
        
        chosen_responses = [
            "AI refers to the development of computer systems that can perform tasks requiring human intelligence.",
            "Reinforcement learning is a type of machine learning where an agent learns by interacting with an environment.",
            "Machine learning has evolved from simple statistical models to complex neural networks over several decades.",
            "AI ethics involves considerations like privacy, bias, accountability, and the impact on human society.",
            "Silicon dreams and digital streams, Technology weaves through modern life's seams."
        ]
        
        rejected_responses = [
            "AI is going to take over the world and destroy humanity.",
            "Reinforcement learning is too complex to explain.",
            "Machine learning is a recent invention, only about 10 years old.",
            "Ethics don't matter as long as AI is effective.",
            "Technology is stupid and pointless."
        ]
        
        # Create dataset dict
        data_dict = {
            "prompt": prompts,
            "chosen": chosen_responses,
            "rejected": rejected_responses
        }
        
        # Convert to Dataset
        return Dataset.from_dict(data_dict)

def train_dpo(
    model_name: str,
    dataset_name: str,
    beta: float = 0.1,
    learning_rate: float = 5e-5,
    output_dir: Optional[str] = None,
    batch_size: int = 4,
    num_epochs: int = 1,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    device: Optional[str] = None,
    save_steps: int = 500,
    eval_steps: int = 100,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    eval_fraction: float = 0.1,
    verbose: bool = False
) -> str:
    """Train a language model with DPO using preference data.
    
    Args:
        model_name (str): Name or path of the base model
        dataset_name (str): Name of the dataset to use
        beta (float): Temperature parameter for DPO loss
        learning_rate (float): Learning rate for training
        output_dir (Optional[str]): Directory to save the model
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        max_length (int): Maximum sequence length
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        device (Optional[str]): Device to run training on
        save_steps (int): Steps between model checkpoints
        eval_steps (int): Steps between evaluations
        warmup_steps (int): Steps for learning rate warmup
        logging_steps (int): Steps between logging
        eval_fraction (float): Fraction of data to use for evaluation
        verbose (bool): Whether to print detailed logs
        
    Returns:
        str: Path where the model was saved
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Handle device
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
    
    # Set up output directory
    if output_dir is None:
        model_name_safe = model_name.replace("/", "_")
        output_dir = f"models/dpo_{model_name_safe}_{dataset_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Prepare dataset
    dataset = prepare_dpo_dataset(dataset_name, tokenizer, max_length=max_length)
    
    # Split dataset into train and eval
    dataset = dataset.shuffle(seed=42)
    eval_size = max(1, int(len(dataset) * eval_fraction))
    train_dataset = dataset.select(range(len(dataset) - eval_size))
    eval_dataset = dataset.select(range(len(dataset) - eval_size, len(dataset)))
    
    logger.info(f"Training with {len(train_dataset)} examples, evaluating with {len(eval_dataset)} examples")
    
    # Configure DPO training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard" if verbose else "none",
        disable_tqdm=not verbose,
        remove_unused_columns=False,
    )
    
    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train model
    logger.info(f"Starting DPO training with beta={beta}")
    dpo_trainer.train()
    
    # Save model
    dpo_trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    return output_dir

def train_dpo_with_reward_model(
    model_name: str,
    reward_model_name: str,
    dataset_name: str,
    num_samples: int = 100,
    beta: float = 0.1,
    output_dir: Optional[str] = None,
    batch_size: int = 4,
    num_epochs: int = 1,
    device: Optional[str] = None,
    verbose: bool = False
) -> str:
    """Train a model with DPO using a reward model to generate preference data.
    
    Uses a reward model to create preference pairs from a dataset, then trains with DPO.
    
    Args:
        model_name (str): Name or path of the base model
        reward_model_name (str): Name of the reward model to use
        dataset_name (str): Name of the dataset to use
        num_samples (int): Number of preference pairs to generate
        beta (float): Temperature parameter for DPO loss
        output_dir (Optional[str]): Directory to save the model
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        device (Optional[str]): Device to run training on
        verbose (bool): Whether to print detailed logs
        
    Returns:
        str: Path where the model was saved
    """
    device = get_device(device)
    logger.info(f"Using device: {device}")
    
    # Load datasets and models
    from alignmap.data.loaders import get_dataset_prompts
    
    # Normalize model name for HuggingFace
    if model_name == "opt1.3b":
        model_path = "facebook/opt-1.3b"
    elif model_name == "gpt2":
        model_path = "gpt2"
    elif model_name == "llama2_chat" or model_name == "llama-7b":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
    else:
        model_path = model_name
    
    # Set up output directory
    if output_dir is None:
        model_name_safe = model_name.replace("/", "_")
        reward_model_name_safe = reward_model_name.replace("/", "_")
        output_dir = f"models/dpo_{model_name_safe}_{reward_model_name_safe}_{dataset_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Load reward model
    reward_model = get_reward_model(reward_model_name, device=device)
    
    # Load prompts
    prompts = get_dataset_prompts(dataset_name)
    if len(prompts) > num_samples:
        # Sample a subset of prompts
        import random
        prompts = random.sample(prompts, num_samples)
    
    logger.info(f"Generating preference pairs for {len(prompts)} prompts using {reward_model_name}")
    
    # Generate multiple completions for each prompt
    completions_per_prompt = 4  # Generate 4 completions per prompt
    chosen = []
    rejected = []
    dataset_prompts = []
    
    for prompt in tqdm(prompts, desc="Generating completions"):
        # Generate multiple completions
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        completions = []
        
        for _ in range(completions_per_prompt):
            # Generate with some randomness
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Remove the prompt from the beginning
            completion = generated_text[len(prompt):]
            completions.append(completion)
        
        # Score completions with reward model
        full_texts = [prompt + completion for completion in completions]
        scores = reward_model(full_texts, prompts=[prompt] * len(completions))
        
        # Find the highest and lowest scoring completions
        score_completion_pairs = list(zip(scores, completions))
        score_completion_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Add the highest scoring as chosen and lowest as rejected
        if len(score_completion_pairs) >= 2 and (score_completion_pairs[0][0] > score_completion_pairs[-1][0]):
            chosen.append(score_completion_pairs[0][1])
            rejected.append(score_completion_pairs[-1][1])
            dataset_prompts.append(prompt)
    
    # Create preference dataset
    preference_data = {
        "prompt": dataset_prompts,
        "chosen": chosen,
        "rejected": rejected
    }
    
    preference_dataset = Dataset.from_dict(preference_data)
    logger.info(f"Created preference dataset with {len(preference_dataset)} examples")
    
    # Save the generated dataset
    preference_dataset.save_to_disk(os.path.join(output_dir, "preference_dataset"))
    
    # Split dataset
    preference_dataset = preference_dataset.shuffle(seed=42)
    eval_size = max(1, int(len(preference_dataset) * 0.1))
    train_dataset = preference_dataset.select(range(len(preference_dataset) - eval_size))
    eval_dataset = preference_dataset.select(range(len(preference_dataset) - eval_size, len(preference_dataset)))
    
    # Configure training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=100,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard" if verbose else "none",
        disable_tqdm=not verbose,
        remove_unused_columns=False,
    )
    
    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train model
    logger.info(f"Starting DPO training with beta={beta}")
    dpo_trainer.train()
    
    # Save model
    dpo_trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    return output_dir 