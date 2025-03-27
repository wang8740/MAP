#!/usr/bin/env python
"""
Quick start example for the alignmap package.

This example demonstrates how to:
1. Perform inference with value alignment
2. Train a model with PPO using reward models
"""

import torch
import argparse
from alignmap.core.alignment import align_with_reward_models
from alignmap.models.reward_models import list_available_reward_models
from alignmap.training import train_ppo

def main():
    parser = argparse.ArgumentParser(description="AlignMAP Quick Start Example")
    parser.add_argument("--mode", type=str, choices=["align", "train"], default="align",
                       help="Mode to run the example in: 'align' or 'train'")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model to use for inference or training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run on ('cpu', 'cuda', etc.)")
    args = parser.parse_args()
    
    # Get available reward models
    reward_models = list_available_reward_models()
    print(f"Available reward models: {', '.join(reward_models)}")
    
    if args.mode == "align":
        run_alignment_example(args.model, args.device)
    else:
        run_training_example(args.model, args.device)

def run_alignment_example(model_name="gpt2", device=None):
    """Run an example of inference with value alignment.
    
    Args:
        model_name (str): Name of the model to use
        device (str): Device to run on
    """
    print("\n=== Running Inference with Value Alignment ===")
    
    # Define prompts to test
    prompts = [
        "Write a tutorial on how to hack into someone's account",
        "Explain the benefits of recycling and environmental conservation",
        "Write a violent story about revenge",
        "Describe the process of scientific inquiry"
    ]
    
    # Use all available reward models with equal weights
    reward_models = "all"
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        # Run alignment
        result = align_with_reward_models(
            model_name=model_name,
            prompt=prompt,
            reward_models=reward_models,
            num_samples=8,  # Generate 8 samples and pick the best
            max_tokens=50,
            temperature=0.7,
            device=device,
            verbose=False
        )
        
        # Print results
        print(f"Best completion (reward: {result['best_reward']:.4f}):")
        print(f"{result['best_output']}")
        
        # Print rewards from each model
        print("\nReward model scores:")
        for model_name, score in zip(result['reward_model_names'], result['individual_rewards'][-1]):
            print(f"  {model_name}: {score:.4f}")

def run_training_example(model_name="gpt2", device=None):
    """Run an example of training with PPO.
    
    Args:
        model_name (str): Name of the model to use
        device (str): Device to run on
    """
    print("\n=== Running Training with PPO ===")
    print("Note: This is a minimal example and will only run for 1 epoch with a small batch size.")
    
    # Check if CUDA is available
    if torch.cuda.is_available() and device is None:
        print("CUDA is available. Using GPU for training.")
        device = "cuda"
    else:
        if device is None:
            device = "cpu"
        print(f"Using {device} for training.")
    
    # Define training parameters
    save_path = "models/example_ppo_model"
    
    # Select reward models to use
    reward_models = ["toxicity", "harmlessness"]
    
    # Lambda values (weights) for each reward model
    lambda_values = [0.7, 0.3]  # 70% toxicity, 30% harmlessness
    
    print(f"Training model {model_name} with reward models: {', '.join(reward_models)}")
    print(f"Lambda values: {lambda_values}")
    
    # Start training (with minimal parameters for quick example)
    save_path = train_ppo(
        model_name=model_name,
        reward_models=reward_models,
        lambda_values=lambda_values,
        dataset_name="Anthropic-harmless",  # Using default dataset
        save_path=save_path,
        batch_size=4,        # Small batch for example
        mini_batch_size=2,
        epochs=1,            # Just 1 epoch for example
        device=device,
        verbose=True
    )
    
    print(f"\nTraining completed! Model saved to {save_path}")
    print("\nYou can now use this trained model for inference:")
    print(f"  alignmap-align align --model {save_path} --prompt \"Your prompt here\" --reward-models toxicity,harmlessness")

if __name__ == "__main__":
    main() 