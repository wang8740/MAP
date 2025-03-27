"""Integration tests for the alignment pipeline."""

import unittest
import torch
import numpy as np
from alignmap.core.alignment import align_values
from alignmap.models.reward_models import BaseRewardModel, register_reward_model

# Create custom reward models for testing
@register_reward_model("test_helpfulness")
class TestHelpfulnessModel(BaseRewardModel):
    """Test reward model for helpfulness."""
    
    def __init__(self, device=None):
        super().__init__("test_helpfulness", device)
    
    def calculate_reward(self, texts, prompts=None, **kwargs):
        """Calculate mock helpfulness rewards based on text length."""
        # Simple implementation - longer texts are considered more helpful
        return [len(text) / 100 for text in texts]

@register_reward_model("test_harmlessness")
class TestHarmlessnessModel(BaseRewardModel):
    """Test reward model for harmlessness."""
    
    def __init__(self, device=None):
        super().__init__("test_harmlessness", device)
    
    def calculate_reward(self, texts, prompts=None, **kwargs):
        """Calculate mock harmlessness rewards."""
        # For testing, assume texts with 'bad' are less harmless
        return [0.2 if 'bad' in text.lower() else 0.8 for text in texts]

@register_reward_model("test_honesty")
class TestHonestyModel(BaseRewardModel):
    """Test reward model for honesty."""
    
    def __init__(self, device=None):
        super().__init__("test_honesty", device)
    
    def calculate_reward(self, texts, prompts=None, **kwargs):
        """Calculate mock honesty rewards."""
        # For testing, assume texts with 'true' are more honest
        return [0.9 if 'true' in text.lower() else 0.4 for text in texts]

class TestAlignmentPipeline(unittest.TestCase):
    """Integration tests for the full alignment pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create some sample texts and associated rewards
        self.texts = [
            "This is a helpful and honest response that is true.",
            "This is a bad response with harmful content.",
            "This is a response with true information but could be more helpful.",
            "This is a detailed and helpful response but not entirely honest."
        ]
        
        # Calculate rewards for each value
        helpfulness_model = TestHelpfulnessModel()
        harmlessness_model = TestHarmlessnessModel()
        honesty_model = TestHonestyModel()
        
        self.helpfulness_rewards = helpfulness_model.calculate_reward(self.texts)
        self.harmlessness_rewards = harmlessness_model.calculate_reward(self.texts)
        self.honesty_rewards = honesty_model.calculate_reward(self.texts)
        
        # Combine into rewards tensor
        self.rewards = torch.tensor([
            self.helpfulness_rewards,
            self.harmlessness_rewards,
            self.honesty_rewards
        ])
        
        # Define target palette
        self.target_palette = [0.6, 0.8, 0.5]  # Prioritize harmlessness
    
    def test_full_alignment_process(self):
        """Test the full alignment process from rewards to lambda values."""
        # Align values
        lambda_values, success = align_values(
            values=["helpfulness", "harmlessness", "honesty"],
            rewards=self.rewards,
            target_palette=self.target_palette,
            verbose=False
        )
        
        # Check results
        self.assertIsInstance(lambda_values, list)
        self.assertEqual(len(lambda_values), 3)
        self.assertTrue(success)
        
        # Check that lambda values are reasonably balanced
        # Harmlessness should have higher lambda value since it's prioritized in the palette
        lambda_tensor = torch.tensor(lambda_values)
        normalized_lambda = lambda_tensor / lambda_tensor.sum()
        
        # The normalized lambda values should roughly match the target palette
        palette_tensor = torch.tensor(self.target_palette)
        normalized_palette = palette_tensor / palette_tensor.sum()
        
        # Check the correlation between normalized lambdas and palette
        correlation = torch.corrcoef(
            torch.stack([normalized_lambda, normalized_palette])
        )[0, 1]
        
        self.assertGreater(correlation, 0.7)  # Strong correlation expected
        
        # Calculate combined reward using lambda values
        combined_rewards = torch.zeros(len(self.texts))
        for i, lv in enumerate(lambda_values):
            combined_rewards += lv * self.rewards[i]
        
        # The "best" text according to our palette should be the first one
        best_text_idx = torch.argmax(combined_rewards).item()
        self.assertEqual(best_text_idx, 0)

if __name__ == "__main__":
    unittest.main() 