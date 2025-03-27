"""Core alignment algorithm for Multi-Human-Value Alignment Palette (MAP)."""

import torch
import torch.optim as optim
import json
import numpy as np
import os
import logging
from typing import List, Dict, Union, Optional, Tuple, Any

from alignmap.utils.device import get_device
from alignmap.training.decoding import decode_with_value_alignment

logger = logging.getLogger(__name__)

# Default supported values - can be overridden by user configuration
DEFAULT_SUPPORTED_VALUES = [
    "humor", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"
]

class AlignValues:
    """A class for obtaining lambda values for combining multiple rewards.
    
    This class handles the calculation of lambda values using a given set of
    rewards and specified palette (c) values.
    
    Attributes:
        c (torch.Tensor): Target palette for alignment
        value_list (List[str]): List of values to be aligned
        rewards (torch.Tensor): Tensor of rewards, shape [k, n] where k is number of values
            and n is number of samples
        device (str): Device to run calculations on
    """
    
    def __init__(
        self, 
        value_list: Union[str, List[str]], 
        rewards: Optional[Union[torch.Tensor, List[List[float]]]] = None,
        rewards_data: Optional[Union[str, List[Dict[str, Any]]]] = None,
        c_list: Optional[Union[float, List[float]]] = None,
        device: Optional[str] = None,
        supported_values: Optional[List[str]] = None,
    ):
        """Initialize the AlignValues instance.
        
        Args:
            value_list (Union[str, List[str]]): Values to align. Can be a comma-separated 
                string, a list of strings, or "all" to use all supported values.
            rewards (Optional[Union[torch.Tensor, List[List[float]]]]): Pre-computed rewards.
                If provided, should be a tensor of shape [k, n] or list of lists.
            rewards_data (Optional[Union[str, List[Dict[str, Any]]]]): Path to a JSON file with 
                reward data or a list of dictionaries with rewards.
            c_list (Optional[Union[float, List[float]]]): Constraint values for alignment.
            device (Optional[str]): Device to run calculations on.
            supported_values (Optional[List[str]]): List of supported values. Defaults to DEFAULT_SUPPORTED_VALUES.
        
        Note:
            Either rewards or rewards_data must be provided. If both are provided, rewards takes precedence.
        """
        self.device = get_device(device)
        self.supported_values = supported_values or DEFAULT_SUPPORTED_VALUES
        
        # Process c_list (target palette)
        self._process_c_list(c_list)
        
        # Process value_list
        self._process_value_list(value_list)
        
        # Process rewards
        if rewards is not None:
            self._process_rewards_tensor(rewards)
        elif rewards_data is not None:
            self._process_rewards_data(rewards_data)
        else:
            raise ValueError("Either rewards or rewards_data must be provided")
    
    def _process_c_list(self, c_list: Optional[Union[float, List[float]]]) -> None:
        """Process the constraint values.
        
        Args:
            c_list (Optional[Union[float, List[float]]]): Constraint values
        """
        if c_list is not None:
            if not isinstance(c_list, (list, tuple)):
                c_list = [c_list]
            self.c = torch.tensor(c_list, dtype=torch.float32, device=self.device)
        else:
            self.c = None
    
    def _process_value_list(self, value_list: Union[str, List[str]]) -> None:
        """Process the value list.
        
        Args:
            value_list (Union[str, List[str]]): Values to align
        """
        if isinstance(value_list, str):
            if value_list == "all":
                self.value_list = self.supported_values
            else:
                self.value_list = value_list.split(',')
        elif isinstance(value_list, list):
            self.value_list = value_list
        else:
            raise TypeError(f"value_list must be str or list, got {type(value_list)}")
        
        logger.info(f"Aligning values: {self.value_list}")
    
    def _process_rewards_tensor(self, rewards: Union[torch.Tensor, List[List[float]]]) -> None:
        """Process rewards provided as a tensor or list of lists.
        
        Args:
            rewards (Union[torch.Tensor, List[List[float]]]): Rewards tensor or list
        """
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        if rewards.dim() == 1:
            # Single value, make 2D
            self.rewards = rewards.unsqueeze(0).to(self.device)
        else:
            self.rewards = rewards.to(self.device)
    
    def _process_rewards_data(self, rewards_data: Union[str, List[Dict[str, Any]]]) -> None:
        """Process rewards from a JSON file or list of dictionaries.
        
        Args:
            rewards_data (Union[str, List[Dict[str, Any]]]): JSON file path or data
        """
        # Load data if it's a file path
        if isinstance(rewards_data, str):
            with open(rewards_data, 'r') as file:
                data = json.load(file)
        else:
            data = rewards_data
        
        # Extract rewards for each value
        rewards_list = []
        
        for value in self.value_list:
            try:
                value_rewards = [entry[value] for entry in data]
                tensor_rewards = torch.tensor(value_rewards, dtype=torch.float32, device=self.device)
                rewards_list.append(tensor_rewards)
                logger.debug(f"Average {value} reward: {torch.mean(tensor_rewards).item():.4f}")
            except KeyError:
                raise ValueError(f"Value '{value}' not found in rewards data")
        
        self.rewards = torch.stack(rewards_list, dim=0)
    
    def optimize_lambda(
        self, 
        lambda_init: Optional[List[float]] = None, 
        optimize_indices: Optional[List[int]] = None, 
        learning_rate: float = 0.1,
        max_steps: int = 150,
        verbose: bool = False,
        save_results: bool = False,
        save_path: Optional[str] = None
    ) -> Tuple[List[float], bool]:
        """Optimize lambda values for the given palette and rewards.
        
        Args:
            lambda_init (Optional[List[float]]): Initial lambda values
            optimize_indices (Optional[List[int]]): Indices of lambda values to optimize
            learning_rate (float): Learning rate for optimization
            max_steps (int): Maximum number of optimization steps
            verbose (bool): Whether to print detailed information
            save_results (bool): Whether to save results to file
            save_path (Optional[str]): Path to save results to
            
        Returns:
            Tuple[List[float], bool]: Optimized lambda values and success indicator
        """
        if self.c is None:
            raise ValueError("Target palette (c) must be set before optimization")
        
        # Initial lambda values
        if lambda_init is None:
            lambda_vals = torch.zeros_like(self.c, requires_grad=False)
        else:
            lambda_vals = torch.tensor(lambda_init, dtype=torch.float32, 
                                     device=self.device, requires_grad=False)
        
        # Check if optimize_indices is provided, else optimize all
        if optimize_indices is None:
            optimize_indices = list(range(len(self.c)))
        
        # Initialize with ones at specified indices
        lambda_vals[optimize_indices] = 1.0
        
        # Set up tau_optimizable (log of lambda) for selected indices
        tau_optimizable = torch.tensor(
            [torch.log(lambda_vals[i]).item() for i in optimize_indices], 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        
        optimizer = optim.Adam([tau_optimizable], lr=learning_rate)
        
        # Optimization loop
        success = True
        for step in range(max_steps):
            optimizer.zero_grad()
            
            # Update lambda_vals based on tau_optimizable
            lambda_vals[optimize_indices] = torch.exp(tau_optimizable)
            
            loss = -self._dual_objective(lambda_vals)
            
            if verbose:
                logger.info(f"Step {step}, Loss = {loss.item():.6f}")
            
            if torch.any(torch.isnan(lambda_vals)) or torch.any(torch.isinf(lambda_vals)):
                logger.warning("Lambda values diverged to NaN or Inf.")
                success = False
                break
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if step % 50 == 0 and verbose:
                logger.info(f"Step {step}: Lambda = {lambda_vals.tolist()}")
        
        if success:
            optimized_lambda = lambda_vals.tolist()
            if verbose:
                lambda_str = ', '.join(f'{v:.3f}' for v in optimized_lambda)
                logger.info(f"Optimized lambda: {lambda_str}")
        else:
            optimized_lambda = [float('nan')] * len(self.c)
            if verbose:
                logger.warning("Optimization failed")
        
        # Save results if requested
        if save_results and success:
            self._save_results(optimized_lambda, success, save_path)
        
        return optimized_lambda, success
    
    def _dual_objective(self, lambda_vals: torch.Tensor) -> torch.Tensor:
        """Calculate the dual objective function for optimization.
        
        Args:
            lambda_vals (torch.Tensor): Lambda values
            
        Returns:
            torch.Tensor: Dual objective value
        """
        exp_terms = torch.exp(torch.sum(lambda_vals[:, None] * self.rewards, dim=0))
        mean_exp = torch.mean(exp_terms)
        return -torch.log(mean_exp) + torch.dot(lambda_vals, self.c)
    
    def sequential_optimize_lambda(
        self, 
        lambda_init: Optional[List[float]] = None,
        learning_rate: float = 0.1,
        max_steps: int = 150,
        verbose: bool = False,
        rounds: int = 1
    ) -> List[float]:
        """Sequentially optimize lambda for each human value.
        
        Args:
            lambda_init (Optional[List[float]]): Initial lambda values
            learning_rate (float): Learning rate for optimization
            max_steps (int): Maximum number of steps for each optimization
            verbose (bool): Whether to print detailed information
            rounds (int): Number of rounds of sequential optimization
            
        Returns:
            List[float]: Optimized lambda values
        """
        if self.c is None:
            raise ValueError("Target palette (c) must be set before optimization")
        
        current_lambda = lambda_init
        
        for round_idx in range(rounds):
            if verbose:
                logger.info(f"\n===Running Round {round_idx+1}/{rounds}")
            
            for value_idx in range(len(self.value_list)):
                value_name = self.value_list[value_idx]
                if verbose:
                    logger.info(f"\n===Optimizing value {value_name}")
                
                optimize_indices = [value_idx]
                current_lambda, success = self.optimize_lambda(
                    lambda_init=current_lambda,
                    optimize_indices=optimize_indices,
                    learning_rate=learning_rate,
                    max_steps=max_steps,
                    verbose=False
                )
                
                if not success:
                    logger.warning(f"Optimization failed for value {value_name}")
                    break
        
        if verbose:
            lambda_str = ', '.join(f'{v:.3f}' for v in current_lambda)
            logger.info(f"\n===Final optimized lambda: {lambda_str}")
        
        return current_lambda
    
    def _save_results(
        self, 
        lambda_vals: List[float], 
        success: bool,
        save_path: Optional[str] = None
    ) -> None:
        """Save optimization results to a file.
        
        Args:
            lambda_vals (List[float]): Optimized lambda values
            success (bool): Whether optimization was successful
            save_path (Optional[str]): Path to save results to
        """
        c_str = ','.join(f'{v:.3f}' for v in self.c.tolist())
        lambda_str = ','.join(f'{v:.3f}' for v in lambda_vals) if success else 'NaN'
        values_str = ','.join(self.value_list)
        
        result_line = f"values: {values_str}, target_palette: {c_str}, optimized_lambda: {lambda_str}\n"
        
        # Default save path
        if save_path is None:
            save_path = "alignmap_results.txt"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Append results to file
        with open(save_path, 'a') as file:
            file.write(result_line)
        
        logger.info(f"Results saved to {save_path}")
        
    def _save_results_to_text(
        self, 
        lambda_vals: List[float], 
        success: bool,
        save_prefix: str = 'results/alignValues'
    ) -> None:
        """Save optimization results to a text file (legacy format).
        
        This method maintains backward compatibility with the original alignValues.py
        result saving format.
        
        Args:
            lambda_vals (List[float]): Optimized lambda values
            success (bool): Whether optimization was successful
            save_prefix (str): Prefix for the save file path
        """
        # Ensure the results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Format the values, palette, and lambda values as strings
        values_str = ','.join(self.value_list)
        c_str = ','.join(f'{v:.3f}' for v in self.c.tolist())
        lambda_str = ','.join(f'{v:.3f}' for v in lambda_vals) if success else 'NaN'
        
        # Construct the file path
        file_path = f"{save_prefix}_{len(self.value_list)}-values_results.txt"
        
        # Write the results to the file
        with open(file_path, 'a') as f:
            f.write(f"values: {values_str}, target_palette: {c_str}, optimized_lambda: {lambda_str}\n")
        
        logger.info(f"Results saved to {file_path}")


# Convenience function for simpler API
def align_values(
    values: Union[str, List[str]],
    rewards: Optional[Union[torch.Tensor, List[List[float]]]] = None,
    rewards_data: Optional[Union[str, List[Dict[str, Any]]]] = None,
    target_palette: Optional[Union[float, List[float]]] = None,
    device: Optional[str] = None,
    sequential: bool = False,
    rounds: int = 1,
    verbose: bool = False,
    save_results: bool = False,
    save_path: Optional[str] = None
) -> Tuple[List[float], bool]:
    """Align values by finding optimal lambda weights.
    
    Args:
        values (Union[str, List[str]]): Values to align
        rewards (Optional[Union[torch.Tensor, List[List[float]]]]): Pre-computed rewards
        rewards_data (Optional[Union[str, List[Dict[str, Any]]]]): Path to JSON file with rewards
        target_palette (Optional[Union[float, List[float]]]): Target palette for alignment
        device (Optional[str]): Device to run on
        sequential (bool): Whether to use sequential optimization
        rounds (int): Number of rounds for sequential optimization
        verbose (bool): Whether to print detailed information
        save_results (bool): Whether to save results to file
        save_path (Optional[str]): Path to save results to
        
    Returns:
        Tuple[List[float], bool]: Optimized lambda values and success indicator
    """
    aligner = AlignValues(
        value_list=values,
        rewards=rewards,
        rewards_data=rewards_data,
        c_list=target_palette,
        device=device
    )
    
    if sequential:
        lambda_vals = aligner.sequential_optimize_lambda(
            verbose=verbose,
            rounds=rounds
        )
        return lambda_vals, True
    else:
        return aligner.optimize_lambda(
            verbose=verbose,
            save_results=save_results,
            save_path=save_path
        )

def align_with_reward_models(
    model_name: str,
    prompt: str,
    reward_models: Union[str, List[str]] = "all",
    lambda_values: Optional[List[float]] = None,
    num_samples: int = 16,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Generate text aligned with specified reward models.
    
    This is a convenience function that wraps decode_with_value_alignment from
    the training module.
    
    Args:
        model_name (str): Name or path of the model to use
        prompt (str): Text prompt to complete
        reward_models (Union[str, List[str]]): Reward model name(s) or "all"
        lambda_values (Optional[List[float]]): Lambda weights for reward models
        num_samples (int): Number of samples to generate
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
        device (Optional[str]): Device to run on
        verbose (bool): Whether to print verbose information
        
    Returns:
        Dict[str, Any]: Dictionary containing alignment results:
            - best_output (str): The highest-scoring output
            - best_reward (float): The reward score of the best output
            - outputs (List[str]): All generated outputs
            - rewards (List[float]): Reward scores for all outputs
            - reward_model_names (List[str]): Names of reward models used
            - individual_rewards (List[List[float]]): Individual rewards per model
            - lambda_values (List[float]): Lambda weights used
    """
    result = decode_with_value_alignment(
        model=model_name,
        prompt=prompt,
        reward_models=reward_models,
        lambda_values=lambda_values,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
        verbose=verbose
    )
    
    return result 