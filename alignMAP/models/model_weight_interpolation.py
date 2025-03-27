"""Utilities for model weight interpolation and merging."""

import os
import logging
from typing import Optional, Union, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

def interpolate_models(
    model_path1: str,
    model_path2: str,
    interpolation_weight: float,
    save_path: Optional[str] = None,
    return_model: bool = False
) -> Optional[Union[PreTrainedModel, tuple]]:
    """Apply linear interpolation between two models.
    
    This function performs model interpolation between two pre-trained models
    based on a given interpolation factor.
    
    Args:
        model_path1 (str): Path to the first pre-trained model
        model_path2 (str): Path to the second pre-trained model
        interpolation_weight (float): Interpolation factor in [0, 1].
                                 0 means full weight to model1, 1 means full weight to model2
        save_path (Optional[str]): Path to save the interpolated model.
                                   If None, the model is not saved
        return_model (bool): Whether to return the interpolated model
    
    Returns:
        Optional[Union[PreTrainedModel, tuple]]: If return_model is True, returns the 
                                                 interpolated model, or a tuple of (model, tokenizer)
                                                 if save_path is not None.
                                                 Otherwise returns None.
    
    Example:
        >>> model_path1 = "modelsDPO/basemodel-1000sample-0.1beta-0.0harmless"
        >>> model_path2 = "modelsDPO/basemodel-1000sample-0.1beta-1.0harmless"
        >>> interpolation_weight = 0.5
        >>> save_path = "soupModel/interpolated_model"
        >>> interpolate_models(model_path1, model_path2, interpolation_weight, save_path)
    """
    logger.info(f"Interpolating models: {model_path1} and {model_path2}")
    logger.info(f"Interpolation weight: {interpolation_weight}")
    
    # Load the models for interpolation
    model1 = AutoModelForCausalLM.from_pretrained(model_path1)
    model2 = AutoModelForCausalLM.from_pretrained(model_path2)
    
    # Load the tokenizer from the first model
    tokenizer = AutoTokenizer.from_pretrained(model_path1)
    
    # Perform model interpolation
    new_model_weights = {}
    with torch.no_grad():  # Use torch.no_grad() for efficient weight interpolation
        for key in model1.state_dict().keys():
            if key in model2.state_dict():
                new_model_weights[key] = (
                    (1 - interpolation_weight) * model1.state_dict()[key] + 
                    interpolation_weight * model2.state_dict()[key]
                )
            else:
                logger.warning(f"Key {key} not found in model2, using model1 weights")
                new_model_weights[key] = model1.state_dict()[key]
    
    # Create a new model instance and load the interpolated weights
    model1.load_state_dict(new_model_weights)
    
    # Save the model if a save path is provided
    if save_path:
        save_model_and_tokenizer(model1, tokenizer, save_path)
        logger.info(f"Interpolated model saved to: {save_path}")
    
    # Return the model if requested
    if return_model:
        if save_path:
            return model1, tokenizer
        else:
            return model1
    
    return None

def save_model_and_tokenizer(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    save_path: str
) -> None:
    """Save a model and tokenizer to disk.
    
    Args:
        model (PreTrainedModel): The model to save
        tokenizer (PreTrainedTokenizer): The tokenizer to save
        save_path (str): Path to save the model and tokenizer
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to: {save_path}")

def multi_model_interpolation(
    base_model_path: str,
    model_paths: List[str],
    weights: List[float],
    save_path: str,
    return_model: bool = False
) -> Optional[Union[PreTrainedModel, tuple]]:
    """Interpolate between multiple models with different weights.
    
    Args:
        base_model_path (str): Path to the base model
        model_paths (List[str]): List of paths to models for interpolation
        weights (List[float]): List of weights for each model (should sum to 1)
        save_path (str): Path to save the interpolated model
        return_model (bool): Whether to return the interpolated model
        
    Returns:
        Optional[Union[PreTrainedModel, tuple]]: Interpolated model or (model, tokenizer) if requested
    
    Raises:
        ValueError: If the number of models doesn't match the number of weights
                   or if weights don't sum to 1
    """
    if len(model_paths) != len(weights):
        raise ValueError("Number of models must match number of weights")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1")
    
    logger.info(f"Multi-model interpolation with {len(model_paths)} models")
    
    # Load the base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Create new weights dictionary starting with zeros
    new_weights = {}
    for key in base_model.state_dict().keys():
        new_weights[key] = torch.zeros_like(base_model.state_dict()[key])
    
    # Interpolate weights from all models
    with torch.no_grad():
        for model_path, weight in zip(model_paths, weights):
            if weight == 0:
                continue  # Skip models with zero weight
                
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            for key in new_weights.keys():
                if key in model.state_dict():
                    new_weights[key] += weight * model.state_dict()[key]
                else:
                    logger.warning(f"Key {key} not found in model {model_path}")
    
    # Load the interpolated weights into the base model
    base_model.load_state_dict(new_weights)
    
    # Save the model if requested
    if save_path:
        save_model_and_tokenizer(base_model, tokenizer, save_path)
        logger.info(f"Multi-interpolated model saved to: {save_path}")
    
    # Return the model if requested
    if return_model:
        if save_path:
            return base_model, tokenizer
        else:
            return base_model
    
    return None 