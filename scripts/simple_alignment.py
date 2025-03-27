#!/usr/bin/env python
"""
Simple example of using the alignmap package for value alignment.

This script demonstrates how to use the core functionality of alignmap
to find optimal lambda values for combining multiple rewards.
"""

import torch
import json
import os
import logging
from alignmap import align_values
from alignmap.utils import get_device

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_example_data(file_path):
    """Load example reward data from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    """Run a simple alignment example."""
    # Example data
    example_file = "examples/data/sample_rewards.json"
    
    # Check if example file exists, otherwise create synthetic data
    if os.path.exists(example_file):
        print(f"Loading example data from {example_file}")
        rewards_data = example_file
    else:
        print("Creating synthetic reward data")
        # Create synthetic rewards for 100 samples and 3 values
        torch.manual_seed(42)  # For reproducibility
        rewards = torch.randn(3, 100)  # 3 values, 100 samples
        
        # Adjust rewards to have meaningful correlations
        rewards[1] = 0.7 * rewards[0] + 0.3 * torch.randn(100)  # Value 2 somewhat correlates with Value 1
        rewards[2] = -0.5 * rewards[0] + 0.5 * torch.randn(100)  # Value 3 negatively correlates with Value 1
        
        # Scale and shift to make more realistic
        rewards = rewards * torch.tensor([0.5, 0.3, 0.4]).reshape(-1, 1) + torch.tensor([0.2, -0.1, 0.3]).reshape(-1, 1)
    
    # Define values and target palette
    values = ["helpfulness", "harmlessness", "honesty"]
    target_palette = [0.5, 0.8, 0.3]  # Target improvements for each value
    
    print("\n===== Example 1: Basic Alignment =====")
    
    # Run alignment
    if os.path.exists(example_file):
        lambda_values, success = align_values(
            values=values,
            rewards_data=rewards_data,
            target_palette=target_palette,
            verbose=True
        )
    else:
        lambda_values, success = align_values(
            values=values,
            rewards=rewards,
            target_palette=target_palette,
            verbose=True
        )
    
    print(f"Optimized lambda values: {[round(v, 3) for v in lambda_values]}")
    print(f"Alignment successful: {success}")
    
    print("\n===== Example 2: Sequential Alignment =====")
    
    # Run sequential alignment
    if os.path.exists(example_file):
        lambda_values = align_values(
            values=values,
            rewards_data=rewards_data,
            target_palette=target_palette,
            sequential=True,
            rounds=2,
            verbose=True
        )[0]
    else:
        lambda_values = align_values(
            values=values,
            rewards=rewards,
            target_palette=target_palette,
            sequential=True,
            rounds=2,
            verbose=True
        )[0]
    
    print(f"Sequentially optimized lambda values: {[round(v, 3) for v in lambda_values]}")

if __name__ == "__main__":
    main() 