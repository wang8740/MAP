"""Plotting utilities for alignment results."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import os

def plot_reward_distribution(
    rewards: Union[List[float], np.ndarray],
    title: str = "Reward Distribution",
    xlabel: str = "Reward Value",
    ylabel: str = "Frequency",
    bins: int = 20,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot the distribution of rewards.
    
    Args:
        rewards (Union[List[float], np.ndarray]): List or array of reward values
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        bins (int): Number of histogram bins
        save_path (Optional[str]): Path to save the figure
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=bins, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_lambda_values(
    lambda_values: List[float],
    value_names: List[str],
    title: str = "Lambda Values",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot the lambda values for each human value.
    
    Args:
        lambda_values (List[float]): Lambda values
        value_names (List[str]): Names of the human values
        title (str): Plot title
        save_path (Optional[str]): Path to save the figure
        show (bool): Whether to display the plot
    """
    if len(lambda_values) != len(value_names):
        raise ValueError(
            f"Number of lambda values ({len(lambda_values)}) must match "
            f"number of value names ({len(value_names)})"
        )
    
    plt.figure(figsize=(10, 6))
    plt.bar(value_names, lambda_values, alpha=0.7, color='green')
    plt.title(title)
    plt.xlabel("Human Values")
    plt.ylabel("Lambda Weight")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_radar_chart(
    values: Dict[str, List[float]],
    categories: List[str],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Create a radar chart for comparing models across multiple values.
    
    Args:
        values (Dict[str, List[float]]): Dictionary mapping model names to values
        categories (List[str]): Names of the categories (human values)
        title (str): Plot title
        save_path (Optional[str]): Path to save the figure
        show (bool): Whether to display the plot
    """
    n_cats = len(categories)
    
    # Create angle for each category
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    for model_name, model_values in values.items():
        # Complete the loop for plotting
        values_plot = model_values.copy()
        values_plot += values_plot[:1]
        
        # Plot values
        ax.plot(angles, values_plot, linewidth=2, label=model_name)
        ax.fill(angles, values_plot, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Add legend and title
    plt.legend(loc='upper right')
    plt.title(title)
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close() 