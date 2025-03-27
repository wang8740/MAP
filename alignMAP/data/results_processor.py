"""Data processing utilities for merging and analyzing MAP results.

This module contains functions for merging data files and processing
reward calculation results to be used in the AlignMAP framework.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def merge_reward_files(filepaths: List[str], output_path: str) -> Dict[str, Any]:
    """Merge multiple reward calculation result files into one consolidated file.
    
    Args:
        filepaths: List of paths to reward calculation result files
        output_path: Path to save the merged results
        
    Returns:
        Dictionary containing the merged data
        
    Raises:
        FileNotFoundError: If any of the specified files do not exist
    """
    # Check if files exist
    for filepath in filepaths:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load and merge data
    merged_data = {}
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Extract filename without extension as a key
                file_key = os.path.splitext(os.path.basename(filepath))[0]
                merged_data[file_key] = data
                logger.info(f"Loaded data from {filepath}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from {filepath}")
    
    # Save merged data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)
    
    logger.info(f"Merged {len(filepaths)} files and saved to {output_path}")
    return merged_data


def process_reward_results(results_file: str) -> Dict[str, np.ndarray]:
    """Process reward calculation results into tensors.
    
    Args:
        results_file: Path to the reward calculation results file
        
    Returns:
        Dictionary mapping reward names to numpy arrays of reward values
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from {results_file}")
    
    # Process data into tensors
    reward_tensors = {}
    
    for reward_name, values in data.items():
        if isinstance(values, list):
            reward_tensors[reward_name] = np.array(values)
        elif isinstance(values, dict) and "values" in values:
            reward_tensors[reward_name] = np.array(values["values"])
    
    logger.info(f"Processed {len(reward_tensors)} reward types from {results_file}")
    return reward_tensors


def extract_rewards_statistics(rewards: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Extract statistics from reward tensors.
    
    Args:
        rewards: Dictionary mapping reward names to numpy arrays of reward values
        
    Returns:
        Dictionary mapping reward names to statistics (min, max, mean, std)
    """
    stats = {}
    
    for reward_name, values in rewards.items():
        if len(values) == 0:
            continue
            
        stats[reward_name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values))
        }
    
    return stats


def convert_rewards_to_dataframe(rewards: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert reward tensors to a pandas DataFrame for easier analysis.
    
    Args:
        rewards: Dictionary mapping reward names to numpy arrays of reward values
        
    Returns:
        DataFrame with rewards as columns
    """
    # Convert to dict of lists for pandas
    data_dict = {name: values.tolist() for name, values in rewards.items()}
    
    # Ensure all arrays have the same length
    max_len = max(len(values) for values in data_dict.values())
    for name, values in data_dict.items():
        if len(values) < max_len:
            # Pad with NaN
            data_dict[name] = values + [np.nan] * (max_len - len(values))
    
    return pd.DataFrame(data_dict) 