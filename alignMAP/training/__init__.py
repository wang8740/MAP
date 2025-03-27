"""Training modules for aligning language models with human values."""

from alignmap.training.ppo import train_ppo
from alignmap.training.dpo import train_dpo, train_dpo_with_reward_model
from alignmap.training.decoding import decode_with_value_alignment

__all__ = [
    "train_ppo",
    "train_dpo",
    "train_dpo_with_reward_model",
    "decode_with_value_alignment"
] 