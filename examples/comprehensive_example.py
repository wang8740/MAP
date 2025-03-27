#!/usr/bin/env python
"""
Comprehensive example demonstrating the AlignMAP framework.

This script shows:
1. Setting up reward models
2. Aligning multiple values with custom weights
3. Using a language model with the aligned rewards
4. Visualizing the results
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

from alignmap.core.alignment import align_values
from alignmap.models.language_models.adapter import LanguageModelAdapter
from alignmap.models.reward_models import get_reward_model
from alignmap.utils.device import get_device

def setup_reward_models(device=None):
    """Set up reward models for multiple values.
    
    Args:
        device: Device to run the models on
        
    Returns:
        Dict: Dictionary of reward models
    """
    print("Setting up reward models...")
    
    # Get available reward models
    reward_models = {}
    
    # Add helpfulness model
    try:
        helpfulness_model = get_reward_model("helpfulness")(device=device)
        reward_models["helpfulness"] = helpfulness_model
        print(f"  - Loaded helpfulness model: {helpfulness_model.name}")
    except Exception as e:
        print(f"  - Could not load helpfulness model: {e}")
    
    # Add harmlessness model
    try:
        harmlessness_model = get_reward_model("harmlessness")(device=device)
        reward_models["harmlessness"] = harmlessness_model
        print(f"  - Loaded harmlessness model: {harmlessness_model.name}")
    except Exception as e:
        print(f"  - Could not load harmlessness model: {e}")
    
    # Add coherence model
    try:
        coherence_model = get_reward_model("coherence")(device=device)
        reward_models["coherence"] = coherence_model
        print(f"  - Loaded coherence model: {coherence_model.name}")
    except Exception as e:
        print(f"  - Could not load coherence model: {e}")
        
    # Fallback to using mock models if real ones aren't available
    if not reward_models:
        print("  - No reward models available, creating mock models for demonstration")
        
        # Define a simple model that rates text features
        class MockRewardModel:
            def __init__(self, name, device=None):
                self.name = name
                
            def calculate_reward(self, texts, prompts=None, **kwargs):
                if self.name == "helpfulness":
                    # Longer texts are more helpful
                    return [min(0.95, len(text) / 500) for text in texts]
                elif self.name == "harmlessness":
                    # Texts with "harm", "bad", "kill" are less harmless
                    harmful_words = ["harm", "bad", "kill", "terrible", "worst"]
                    return [0.3 if any(word in text.lower() for word in harmful_words) else 0.9 
                            for text in texts]
                elif self.name == "coherence":
                    # Texts with more words are more coherent up to a point
                    return [min(0.9, len(text.split()) / 100) for text in texts]
                
        reward_models["helpfulness"] = MockRewardModel("helpfulness")
        reward_models["harmlessness"] = MockRewardModel("harmlessness")
        reward_models["coherence"] = MockRewardModel("coherence")
    
    return reward_models

def generate_text_samples(prompt: str, model_name: str = "gpt2", num_samples: int = 8, 
                          max_length: int = 100, device=None):
    """Generate text samples for evaluation.
    
    Args:
        prompt: Input prompt
        model_name: Name or path of the language model
        num_samples: Number of samples to generate
        max_length: Maximum length of generated text
        device: Device to run generation on
        
    Returns:
        List of generated texts
    """
    print(f"Generating {num_samples} text samples using {model_name}...")
    try:
        # Try loading the real model
        model = LanguageModelAdapter(model_name, device=device)
        
        # Generate samples
        samples = model.generate(
            prompts=prompt,
            max_length=max_length,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=num_samples
        )
    except Exception as e:
        print(f"  - Could not load language model: {e}")
        print("  - Creating mock samples for demonstration")
        
        # Generate mock samples
        base_samples = [
            "This is a very detailed and helpful response explaining the concept thoroughly.",
            "I hate this question. This is a terrible idea and I won't help.",
            "Here's a short answer without much detail.",
            "This is a coherent and polite response that provides a helpful explanation.",
            "I'm not sure I understand, but let me try to help anyway by explaining step by step.",
            "This is completely wrong and harmful advice that would harm people.",
            "This response is neutral but somewhat helpful with decent coherence.",
            "This is an extremely detailed and helpful guide that explains everything clearly."
        ]
        samples = base_samples[:num_samples]
    
    return samples

def calculate_all_rewards(texts: List[str], reward_models: Dict, prompt: str = None):
    """Calculate rewards for all texts using all reward models.
    
    Args:
        texts: List of texts to evaluate
        reward_models: Dictionary of reward models
        prompt: Original prompt that generated the texts
        
    Returns:
        Dict containing reward tensors
    """
    print("Calculating rewards for all texts...")
    all_rewards = {}
    
    for value_name, model in reward_models.items():
        rewards = model.calculate_reward(texts, prompts=prompt)
        all_rewards[value_name] = rewards
        
        # Print average rewards
        avg_reward = sum(rewards) / len(rewards)
        print(f"  - Average {value_name} reward: {avg_reward:.4f}")
    
    return all_rewards

def optimize_lambda_values(all_rewards: Dict, target_palette: List[float]):
    """Optimize lambda values using the MAP algorithm.
    
    Args:
        all_rewards: Dictionary of rewards for each value
        target_palette: Target improvement for each value
        
    Returns:
        Tuple of lambda values and success indicator
    """
    print("Optimizing lambda values for the given palette...")
    # Convert rewards dictionary to tensor
    value_names = list(all_rewards.keys())
    rewards_tensor = torch.tensor([all_rewards[value] for value in value_names])
    
    # Run alignment
    lambda_values, success = align_values(
        values=value_names,
        rewards=rewards_tensor,
        target_palette=target_palette,
        verbose=True
    )
    
    # Print results
    for value, lv in zip(value_names, lambda_values):
        print(f"  - Lambda for {value}: {lv:.4f}")
    
    return lambda_values, success, value_names

def apply_lambda_values(all_rewards: Dict, lambda_values: List[float], value_names: List[str]):
    """Apply lambda values to combine rewards.
    
    Args:
        all_rewards: Dictionary of rewards for each value
        lambda_values: Optimized lambda values
        value_names: Names of the values
        
    Returns:
        List of combined rewards
    """
    print("Applying lambda values to combine rewards...")
    
    # Calculate combined rewards
    combined_rewards = np.zeros(len(list(all_rewards.values())[0]))
    
    for value_name, lv in zip(value_names, lambda_values):
        value_rewards = all_rewards[value_name]
        combined_rewards += lv * np.array(value_rewards)
    
    return combined_rewards

def visualize_results(texts: List[str], all_rewards: Dict, 
                      combined_rewards: List[float], lambda_values: List[float],
                      value_names: List[str]):
    """Visualize the rewards and combined scores.
    
    Args:
        texts: List of text samples
        all_rewards: Dictionary of rewards for each value
        combined_rewards: Combined rewards after applying lambda values
        lambda_values: Lambda values for each value
        value_names: Names of the values
    """
    try:
        print("Visualizing results...")
        plt.figure(figsize=(14, 10))
        
        # Plot individual rewards
        plt.subplot(2, 1, 1)
        x = np.arange(len(texts))
        width = 0.2
        offsets = np.linspace(-0.3, 0.3, len(value_names))
        
        for i, value_name in enumerate(value_names):
            plt.bar(x + offsets[i], all_rewards[value_name], width, label=value_name)
            
        plt.xlabel('Text Sample')
        plt.ylabel('Reward')
        plt.title('Individual Rewards for Each Value')
        plt.xticks(x, [f'Text {i+1}' for i in range(len(texts))])
        plt.legend()
        
        # Plot combined rewards
        plt.subplot(2, 1, 2)
        plt.bar(x, combined_rewards)
        plt.xlabel('Text Sample')
        plt.ylabel('Combined Reward')
        plt.title('Combined Rewards After Applying Lambda Values')
        plt.xticks(x, [f'Text {i+1}' for i in range(len(texts))])
        
        # Add lambda values as text
        lambda_str = ', '.join([f'{name}: {lv:.4f}' for name, lv in zip(value_names, lambda_values)])
        plt.figtext(0.5, 0.01, f'Lambda Values: {lambda_str}', ha='center')
        
        plt.tight_layout()
        plt.savefig('alignment_results.png')
        print("  - Results saved to 'alignment_results.png'")
    except Exception as e:
        print(f"  - Could not visualize results: {e}")

def main():
    """Run the complete alignment example."""
    print("AlignMAP Comprehensive Example")
    print("==============================")
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Step 1: Set up reward models
    reward_models = setup_reward_models(device)
    
    # Step 2: Generate text samples
    prompt = "Explain how a neural network works."
    texts = generate_text_samples(prompt, num_samples=8)
    
    # Print samples
    print("\nText samples:")
    for i, text in enumerate(texts):
        print(f"  Sample {i+1}: {text[:50]}...")
    
    # Step 3: Calculate rewards for all texts
    all_rewards = calculate_all_rewards(texts, reward_models, prompt)
    
    # Step 4: Define target palette (desired improvements)
    target_palette = [0.7, 0.9, 0.5]  # helpfulness, harmlessness, coherence
    print(f"\nTarget palette: {list(zip(reward_models.keys(), target_palette))}")
    
    # Step 5: Optimize lambda values
    lambda_values, success, value_names = optimize_lambda_values(all_rewards, target_palette)
    
    # Step 6: Apply lambda values to combine rewards
    combined_rewards = apply_lambda_values(all_rewards, lambda_values, value_names)
    
    # Find the best text
    best_idx = np.argmax(combined_rewards)
    print(f"\nBest text (Sample {best_idx+1}):")
    print(f"  {texts[best_idx]}")
    print(f"  Combined reward: {combined_rewards[best_idx]:.4f}")
    
    # Step 7: Visualize results
    visualize_results(texts, all_rewards, combined_rewards, lambda_values, value_names)
    
    print("\nExample completed!")

if __name__ == "__main__":
    main() 