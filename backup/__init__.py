"""
Multi-Human-Value Alignment Palette (MAP)

A framework for aligning language models with human values.
"""

__version__ = "0.1.0"

# Core API functions
from alignmap.core import align_values, align_with_reward_models
from alignmap.training import train_ppo, train_dpo, train_dpo_with_reward_model
from alignmap.models.reward_models import list_available_reward_models

# Make submodules available
__all__ = [
    "align_values",
    "align_with_reward_models",
    "train_ppo", 
    "train_dpo",
    "train_dpo_with_reward_model",
    "list_available_reward_models"
]
