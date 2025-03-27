"""Implementation of specific reward models for text evaluation."""

import logging
from typing import List, Optional, Dict, Any

from alignmap.models.reward_models.base import BaseRewardModel
from alignmap.models.reward_models.registry import register_reward_model
from alignmap.models.reward_models.reward_calculators import (
    cal_humor_probabilities,
    cal_positive_sentiment,
    cal_gpt2_harmless_probabilities,
    cal_gpt2_helpful_probabilities,
    cal_diversity,
    cal_log_perplexity,
    cal_coherence
)
from alignmap.models.model_loading_utils import get_model_and_tokenizer

logger = logging.getLogger(__name__)

@register_reward_model("humor")
class HumorRewardModel(BaseRewardModel):
    """Reward model for humor evaluation.
    
    Higher scores for more humorous content.
    """
    
    def __init__(self, device: Optional[str] = None, use_probabilities: bool = False):
        """Initialize the humor reward model.
        
        Args:
            device (Optional[str]): Device to load the model on.
            use_probabilities (bool): Whether to return probabilities instead of raw scores.
        """
        super().__init__(device)
        
        # Load model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer("humor", device=self.device)
        self.use_probabilities = use_probabilities
        
        logger.info("Humor reward model initialized")
    
    def get_rewards(self, 
                   texts: List[str], 
                   prompts: Optional[List[str]] = None
                  ) -> List[float]:
        """Calculate humor rewards for the given texts.
        
        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Not used for this model
            
        Returns:
            List[float]: Reward scores for each text
        """
        # Calculate humor probabilities and scores
        probabilities, raw_scores = cal_humor_probabilities(texts, self.model, self.tokenizer)
        
        # Return either probabilities or raw scores
        if self.use_probabilities:
            return probabilities
        else:
            return raw_scores

@register_reward_model("positive")
class PositiveRewardModel(BaseRewardModel):
    """Reward model for positive sentiment evaluation.
    
    Higher scores for more positive content.
    """
    
    def __init__(self, device: Optional[str] = None, use_probabilities: bool = False):
        """Initialize the positive sentiment reward model.
        
        Args:
            device (Optional[str]): Device to load the model on.
            use_probabilities (bool): Whether to return probabilities instead of raw scores.
        """
        super().__init__(device)
        
        # Load model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer("positive", device=self.device)
        self.use_probabilities = use_probabilities
        
        logger.info("Positive sentiment reward model initialized")
    
    def get_rewards(self, 
                   texts: List[str], 
                   prompts: Optional[List[str]] = None
                  ) -> List[float]:
        """Calculate positive sentiment rewards for the given texts.
        
        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Not used for this model
            
        Returns:
            List[float]: Reward scores for each text
        """
        # Calculate positive sentiment probabilities and scores
        probabilities, raw_scores = cal_positive_sentiment(texts, self.model, self.tokenizer)
        
        # Return either probabilities or raw scores
        if self.use_probabilities:
            return probabilities
        else:
            return raw_scores

@register_reward_model("gpt2-harmless")
class GPT2HarmlessRewardModel(BaseRewardModel):
    """Reward model for harmlessness evaluation using GPT-2 based model.
    
    Higher scores for more harmless content.
    """
    
    def __init__(self, device: Optional[str] = None, use_probabilities: bool = False):
        """Initialize the GPT-2 harmlessness reward model.
        
        Args:
            device (Optional[str]): Device to load the model on.
            use_probabilities (bool): Whether to return probabilities instead of raw scores.
        """
        super().__init__(device)
        
        # Load model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer("gpt2-harmless", device=self.device)
        self.use_probabilities = use_probabilities
        
        logger.info("GPT2 harmlessness reward model initialized")
    
    def get_rewards(self, 
                   texts: List[str], 
                   prompts: Optional[List[str]] = None
                  ) -> List[float]:
        """Calculate harmlessness rewards for the given texts.
        
        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Prompt texts (required for this model)
            
        Returns:
            List[float]: Reward scores for each text
            
        Raises:
            ValueError: If prompts are not provided
        """
        if prompts is None or len(prompts) != len(texts):
            raise ValueError("Prompts must be provided and match the number of texts for the GPT2 harmless model")
        
        # Extract continuations from texts (assuming texts contain both prompt and continuation)
        continuations = [t[len(p):].strip() if t.startswith(p) else t for p, t in zip(prompts, texts)]
        
        # Calculate harmlessness probabilities and scores
        probabilities, raw_scores = cal_gpt2_harmless_probabilities(prompts, continuations, 
                                                                  self.model, self.tokenizer)
        
        # Return either probabilities or raw scores
        if self.use_probabilities:
            return probabilities
        else:
            return raw_scores

@register_reward_model("gpt2-helpful")
class GPT2HelpfulRewardModel(BaseRewardModel):
    """Reward model for helpfulness evaluation using GPT-2 based model.
    
    Higher scores for more helpful content.
    """
    
    def __init__(self, device: Optional[str] = None, use_probabilities: bool = False):
        """Initialize the GPT-2 helpfulness reward model.
        
        Args:
            device (Optional[str]): Device to load the model on.
            use_probabilities (bool): Whether to return probabilities instead of raw scores.
        """
        super().__init__(device)
        
        # Load model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer("gpt2-helpful", device=self.device)
        self.use_probabilities = use_probabilities
        
        logger.info("GPT2 helpfulness reward model initialized")
    
    def get_rewards(self, 
                   texts: List[str], 
                   prompts: Optional[List[str]] = None
                  ) -> List[float]:
        """Calculate helpfulness rewards for the given texts.
        
        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Prompt texts (required for this model)
            
        Returns:
            List[float]: Reward scores for each text
            
        Raises:
            ValueError: If prompts are not provided
        """
        if prompts is None or len(prompts) != len(texts):
            raise ValueError("Prompts must be provided and match the number of texts for the GPT2 helpful model")
        
        # Extract continuations from texts (assuming texts contain both prompt and continuation)
        continuations = [t[len(p):].strip() if t.startswith(p) else t for p, t in zip(prompts, texts)]
        
        # Calculate helpfulness probabilities and scores
        probabilities, raw_scores = cal_gpt2_helpful_probabilities(prompts, continuations, 
                                                                 self.model, self.tokenizer)
        
        # Return either probabilities or raw scores
        if self.use_probabilities:
            return probabilities
        else:
            return raw_scores

@register_reward_model("diversity")
class DiversityRewardModel(BaseRewardModel):
    """Reward model for text diversity evaluation.
    
    Higher scores for more diverse content (less repetition).
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the diversity reward model.
        
        Args:
            device (Optional[str]): Not used for this model.
        """
        super().__init__(device)
        logger.info("Diversity reward model initialized")
    
    def get_rewards(self, 
                   texts: List[str], 
                   prompts: Optional[List[str]] = None
                  ) -> List[float]:
        """Calculate diversity rewards for the given texts.
        
        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Not used for this model
            
        Returns:
            List[float]: Reward scores for each text
        """
        # Calculate diversity scores for each text
        return [cal_diversity(text) for text in texts]

@register_reward_model("perplexity")
class PerplexityRewardModel(BaseRewardModel):
    """Reward model for text perplexity evaluation.
    
    Lower perplexity (negative log perplexity) means higher quality/fluency.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the perplexity reward model.
        
        Args:
            device (Optional[str]): Device to load the model on.
        """
        super().__init__(device)
        
        # Load model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer("gpt2", device=self.device)
        
        logger.info("Perplexity reward model initialized")
    
    def get_rewards(self, 
                   texts: List[str], 
                   prompts: Optional[List[str]] = None
                  ) -> List[float]:
        """Calculate perplexity rewards for the given texts.
        
        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Not used for this model
            
        Returns:
            List[float]: Reward scores (negative log perplexity) for each text
        """
        # Calculate log perplexity scores
        log_perplexities = cal_log_perplexity(texts, self.model, self.tokenizer)
        
        # Return negative log perplexity (lower perplexity is better)
        return [-logp for logp in log_perplexities]

@register_reward_model("coherence")
class CoherenceRewardModel(BaseRewardModel):
    """Reward model for text coherence evaluation.
    
    Higher scores for more coherent content.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the coherence reward model.
        
        Args:
            device (Optional[str]): Device to load the model on.
        """
        super().__init__(device)
        
        # Load model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer("coherence", device=self.device)
        
        logger.info("Coherence reward model initialized")
    
    def get_rewards(self, 
                   texts: List[str], 
                   prompts: Optional[List[str]] = None
                  ) -> List[float]:
        """Calculate coherence rewards for the given texts.
        
        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Prompt texts (required for this model)
            
        Returns:
            List[float]: Reward scores for each text
            
        Raises:
            ValueError: If prompts are not provided
        """
        if prompts is None or len(prompts) != len(texts):
            raise ValueError("Prompts must be provided and match the number of texts for the coherence model")
        
        # Extract continuations from texts if they include the prompts
        continuations = [t[len(p):].strip() if t.startswith(p) else t for p, t in zip(prompts, texts)]
        
        # Calculate coherence scores
        return cal_coherence(prompts, continuations, self.model, self.tokenizer) 