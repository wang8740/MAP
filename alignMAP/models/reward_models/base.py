"""Base classes for reward models in alignmap."""

import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable

class BaseRewardModel(ABC):
    """Base class for reward models.
    
    All reward models should inherit from this class and implement the
    required methods for compatibility with the alignmap framework.
    """
    
    def __init__(self, name: str, device: Optional[str] = None):
        """Initialize a reward model.
        
        Args:
            name (str): Name of the reward model
            device (Optional[str]): Device to run the model on
        """
        self.name = name
        self.device = device
    
    @abstractmethod
    def calculate_reward(
        self, 
        texts: List[str], 
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """Calculate rewards for the given texts.
        
        Args:
            texts (List[str]): Texts to calculate rewards for
            prompts (Optional[List[str]]): Prompts that generated the texts
            **kwargs: Additional arguments specific to the reward model
            
        Returns:
            List[float]: Rewards for each text
        """
        pass
    
    def batch_calculate_reward(
        self,
        texts: List[str],
        prompts: Optional[List[str]] = None,
        batch_size: int = 32,
        **kwargs
    ) -> List[float]:
        """Calculate rewards in batches.
        
        Args:
            texts (List[str]): Texts to calculate rewards for
            prompts (Optional[List[str]]): Prompts that generated the texts
            batch_size (int): Size of batches for processing
            **kwargs: Additional arguments specific to the reward model
            
        Returns:
            List[float]: Rewards for each text
        """
        # Process in batches for memory efficiency
        all_rewards = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_prompts = None
            if prompts:
                batch_prompts = prompts[i:i + batch_size]
            
            batch_rewards = self.calculate_reward(
                batch_texts, 
                prompts=batch_prompts,
                **kwargs
            )
            
            all_rewards.extend(batch_rewards)
        
        return all_rewards
    
    def __call__(
        self, 
        texts: Union[str, List[str]], 
        prompts: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Union[float, List[float]]:
        """Make the reward model callable.
        
        Args:
            texts (Union[str, List[str]]): Text(s) to calculate rewards for
            prompts (Optional[Union[str, List[str]]]): Prompt(s) that generated the text(s)
            **kwargs: Additional arguments specific to the reward model
            
        Returns:
            Union[float, List[float]]: Reward(s) for the text(s)
        """
        # Handle single text
        single_input = isinstance(texts, str)
        
        if single_input:
            texts = [texts]
            prompts = [prompts] if prompts is not None else None
        
        rewards = self.calculate_reward(texts, prompts=prompts, **kwargs)
        
        return rewards[0] if single_input else rewards


class FunctionRewardModel(BaseRewardModel):
    """Reward model that uses a function to calculate rewards.
    
    This class allows using custom functions as reward models without
    having to implement a full class.
    """
    
    def __init__(
        self, 
        name: str, 
        reward_fn: Callable[[List[str], Optional[List[str]], Dict[str, Any]], List[float]],
        device: Optional[str] = None
    ):
        """Initialize a function-based reward model.
        
        Args:
            name (str): Name of the reward model
            reward_fn (Callable): Function that calculates rewards
            device (Optional[str]): Device to run the model on
        """
        super().__init__(name, device)
        self.reward_fn = reward_fn
    
    def calculate_reward(
        self, 
        texts: List[str], 
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """Calculate rewards using the provided function.
        
        Args:
            texts (List[str]): Texts to calculate rewards for
            prompts (Optional[List[str]]): Prompts that generated the texts
            **kwargs: Additional arguments passed to the reward function
            
        Returns:
            List[float]: Rewards for each text
        """
        return self.reward_fn(texts, prompts, kwargs) 