"""Unit tests for reward models."""

import unittest
from unittest.mock import MagicMock, patch
import torch

from alignmap.models.reward_models import (
    BaseRewardModel, 
    register_reward_model, 
    get_reward_model,
    list_reward_models
)

class TestRewardModelRegistry(unittest.TestCase):
    """Tests for the reward model registry system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test reward model class
        @register_reward_model("test_reward")
        class TestRewardModel(BaseRewardModel):
            def __init__(self, device=None):
                super().__init__("test_reward", device)
                
            def calculate_reward(self, texts, prompts=None, **kwargs):
                # Simple mock implementation
                return [0.5] * len(texts)
        
        self.TestRewardModel = TestRewardModel
    
    def test_register_and_get_reward_model(self):
        """Test registering and retrieving a reward model."""
        # Get the registered model
        model_class = get_reward_model("test_reward")
        
        # Check it's the right class
        self.assertEqual(model_class, self.TestRewardModel)
        
        # Instantiate the model
        model = model_class()
        
        # Check it's an instance of BaseRewardModel
        self.assertIsInstance(model, BaseRewardModel)
        
        # Check the name is set correctly
        self.assertEqual(model.name, "test_reward")
    
    def test_list_reward_models(self):
        """Test listing registered reward models."""
        models = list_reward_models()
        
        # Check our test model is in the list
        self.assertIn("test_reward", models)
    
    def test_calculate_reward(self):
        """Test calculating rewards with a model."""
        model = self.TestRewardModel()
        
        # Calculate rewards for some texts
        texts = ["Hello, world!", "How are you?"]
        rewards = model.calculate_reward(texts)
        
        # Check results
        self.assertEqual(len(rewards), len(texts))
        self.assertEqual(rewards, [0.5, 0.5])

class TestBaseRewardModel(unittest.TestCase):
    """Tests for the BaseRewardModel class."""
    
    def test_abstract_method(self):
        """Test that subclasses must implement calculate_reward."""
        # Attempting to instantiate a direct subclass without implementing
        # calculate_reward should raise NotImplementedError
        class IncompleteRewardModel(BaseRewardModel):
            def __init__(self):
                super().__init__("incomplete")
        
        model = IncompleteRewardModel()
        
        with self.assertRaises(NotImplementedError):
            model.calculate_reward(["test"])
    
    def test_batch_calculate_reward(self):
        """Test batch calculation of rewards."""
        # Create a concrete subclass
        class ConcreteRewardModel(BaseRewardModel):
            def __init__(self):
                super().__init__("concrete")
                
            def calculate_reward(self, texts, prompts=None, **kwargs):
                # Simple implementation that returns text length as reward
                return [len(text) for text in texts]
        
        model = ConcreteRewardModel()
        
        # Calculate rewards in batches
        texts = ["a", "ab", "abc", "abcd", "abcde"]
        batch_size = 2
        
        rewards = model.batch_calculate_reward(texts, batch_size=batch_size)
        
        # Check results
        self.assertEqual(rewards, [1, 2, 3, 4, 5])

if __name__ == "__main__":
    unittest.main() 