#!/usr/bin/env python
"""
Command-line script to align values using the AlignMAP framework.

Usage:
    python align_values_cmd.py --values helpfulness,harmlessness --palette 0.7,0.9 --rewards-file data.json
    
    or with rewards directly:
    python align_values_cmd.py --values helpfulness,harmlessness --palette 0.7,0.9 --rewards [[0.8,0.6],[0.3,0.7]]
"""

import argparse
import json
import sys
import torch
import numpy as np
from typing import List, Dict, Any

from alignmap.core.alignment import align_values

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Align values using the AlignMAP framework")
    
    # Required arguments
    parser.add_argument("--values", type=str, required=True,
                        help="Comma-separated list of values to align, or 'all'")
    parser.add_argument("--palette", type=str, required=True,
                        help="Comma-separated list of target palette values")
    
    # Optional arguments for rewards
    reward_group = parser.add_argument_group("Rewards")
    reward_group.add_argument("--rewards", type=str, 
                             help="JSON string of rewards array, shape [k, n]")
    reward_group.add_argument("--rewards-file", type=str,
                             help="Path to JSON file with rewards data")
    
    # Other options
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential optimization")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds for sequential optimization")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--save", action="store_true",
                        help="Save results to file")
    parser.add_argument("--output", type=str, default="alignment_results.json",
                        help="Output file for results")
                        
    args = parser.parse_args()
    
    # Validate args
    if args.rewards is None and args.rewards_file is None:
        parser.error("Either --rewards or --rewards-file must be provided")
    
    return args

def process_values(values_str: str) -> List[str]:
    """Process values string into a list."""
    if values_str == "all":
        return "all"
    return values_str.split(",")

def process_palette(palette_str: str) -> List[float]:
    """Process palette string into a list of floats."""
    try:
        return [float(c) for c in palette_str.split(",")]
    except ValueError:
        print("Error: Palette must be comma-separated floats")
        sys.exit(1)

def process_rewards(rewards_str: str) -> List[List[float]]:
    """Process rewards string into a list of lists."""
    try:
        return json.loads(rewards_str)
    except json.JSONDecodeError:
        print("Error: Rewards must be valid JSON")
        sys.exit(1)

def save_results(lambda_values: List[float], success: bool, values: List[str], 
                output_file: str):
    """Save alignment results to file."""
    results = {
        "lambda_values": {value: lv for value, lv in zip(values, lambda_values)},
        "success": success,
        "normalized_lambda": {
            value: lv / sum(lambda_values) 
            for value, lv in zip(values, lambda_values)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    """Run alignment from command line arguments."""
    args = parse_arguments()
    
    # Process arguments
    values = process_values(args.values)
    palette = process_palette(args.palette)
    
    # Get rewards
    if args.rewards:
        rewards = process_rewards(args.rewards)
        rewards_data = None
    else:
        rewards = None
        rewards_data = args.rewards_file
    
    # Run alignment
    try:
        lambda_values, success = align_values(
            values=values,
            rewards=rewards,
            rewards_data=rewards_data,
            target_palette=palette,
            sequential=args.sequential,
            rounds=args.rounds,
            verbose=args.verbose,
            save_results=args.save,
            save_path=args.output if args.save else None
        )
        
        # Print results
        value_list = values if isinstance(values, list) else values.split(",")
        print("\nAlignment Results:")
        print("-----------------")
        print(f"Success: {success}")
        print("\nLambda values:")
        for value, lv in zip(value_list, lambda_values):
            print(f"  {value}: {lv:.6f}")
        
        print("\nNormalized lambda values:")
        lambda_sum = sum(lambda_values)
        for value, lv in zip(value_list, lambda_values):
            print(f"  {value}: {(lv / lambda_sum):.6f}")
        
        # Save results if not already saved through align_values
        if not args.save and args.output:
            save_results(lambda_values, success, value_list, args.output)
            
    except Exception as e:
        print(f"Error during alignment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 