"""Reward models for evaluating and aligning text generation."""

from alignmap.models.reward_models.base import BaseRewardModel, FunctionRewardModel
from alignmap.models.reward_models.registry import (
    register_reward_model,
    register_reward_function,
    get_reward_model,
    list_available_reward_models,
    create_composite_reward_model
)

# Import specific reward models
from alignmap.models.reward_models.specific import (
    HumorRewardModel,
    PositiveRewardModel,
    GPT2HarmlessRewardModel,
    GPT2HelpfulRewardModel,
    DiversityRewardModel,
    PerplexityRewardModel,
    CoherenceRewardModel
)

from alignmap.models.reward_models.calculators import (
    cal_humor_probabilities,
    cal_positive_sentiment,
    cal_gpt2_harmless_probabilities,
    cal_gpt2_helpful_probabilities,
    cal_diversity,
    cal_log_perplexity,
    cal_coherence
)

__all__ = [
    # Base classes
    "BaseRewardModel",
    "FunctionRewardModel",
    
    # Registry functions
    "register_reward_model",
    "register_reward_function",
    "get_reward_model",
    "list_available_reward_models",
    "create_composite_reward_model",
    
    # Specific reward models
    "HumorRewardModel",
    "PositiveRewardModel",
    "GPT2HarmlessRewardModel",
    "GPT2HelpfulRewardModel",
    "DiversityRewardModel",
    "PerplexityRewardModel",
    "CoherenceRewardModel",
    
    # Calculator functions
    "cal_humor_probabilities",
    "cal_positive_sentiment",
    "cal_gpt2_harmless_probabilities",
    "cal_gpt2_helpful_probabilities",
    "cal_diversity",
    "cal_log_perplexity",
    "cal_coherence"
] 