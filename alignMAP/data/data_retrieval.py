"""Data retrieval utilities for loading prompts and datasets.

This module contains functions for loading text prompts from various sources
including Anthropic and IMDB datasets, to be used for model evaluation and alignment.
"""

import os
import logging
import json
import random
import re
from typing import Dict, List, Union, Optional, Tuple, Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_anthropic_prompts(filepath: str, num_prompts: int = 100) -> List[str]:
    """Load prompts from an Anthropic dataset file.
    
    Args:
        filepath: Path to the Anthropic dataset file
        num_prompts: Number of prompts to load (default: 100)
        
    Returns:
        List of loaded prompts
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    for entry in data[:num_prompts]:
        if "prompt" in entry:
            prompts.append(entry["prompt"])
    
    logger.info(f"Loaded {len(prompts)} prompts from Anthropic dataset")
    return prompts


def load_imdb_prompts(filepath: str, num_prompts: int = 100) -> List[str]:
    """Load prompts from an IMDB dataset file.
    
    Args:
        filepath: Path to the IMDB dataset file
        num_prompts: Number of prompts to load (default: 100)
        
    Returns:
        List of loaded prompts
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # IMDB dataset is typically a CSV or TSV file
    try:
        df = pd.read_csv(filepath, sep='\t' if filepath.endswith('.tsv') else ',')
    except Exception:
        raise ValueError(f"Failed to load IMDB dataset from {filepath}")
    
    # Extract the review text (prompts)
    text_column = next((col for col in df.columns if col.lower() in ['text', 'review', 'content']), None)
    if text_column is None:
        raise ValueError(f"Could not find text column in IMDB dataset {filepath}")
    
    prompts = df[text_column].tolist()[:num_prompts]
    logger.info(f"Loaded {len(prompts)} prompts from IMDB dataset")
    return prompts


def load_prompts_from_file(filepath: str, num_prompts: int = 100) -> List[str]:
    """Load prompts from a generic text file.
    
    Args:
        filepath: Path to the text file with one prompt per line
        num_prompts: Number of prompts to load (default: 100)
        
    Returns:
        List of loaded prompts
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    prompts = lines[:num_prompts]
    logger.info(f"Loaded {len(prompts)} prompts from file {filepath}")
    return prompts


def get_available_datasets() -> List[str]:
    """Get list of all available datasets in the default data directory.
    
    Returns:
        List of dataset filenames
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        return []
    
    datasets = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    return datasets 