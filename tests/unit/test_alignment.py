"""Unit tests for the alignment module."""

import unittest
import torch
import numpy as np
from alignmap.core.alignment import AlignValues, align_values

class TestAlignValues(unittest.TestCase):
    """Tests for the AlignValues class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock rewards for 3 values across 5 samples
        self.rewards = torch.tensor([
            [0.8, 0.6, 0.7, 0.9, 0.5],  # Value 1 rewards
            [0.3, 0.7, 0.2, 0.1, 0.8],  # Value 2 rewards
            [0.5, 0.4, 0.9, 0.6, 0.3]   # Value 3 rewards
        ])
        self.values = ["helpfulness", "harmlessness", "honesty"]
        self.c_list = [0.5, 0.3, 0.2]  # Target palette
    
    def test_initialization(self):
        """Test initialization of AlignValues."""
        align = AlignValues(
            value_list=self.values,
            rewards=self.rewards,
            c_list=self.c_list
        )
        
        # Check attributes
        self.assertEqual(align.value_list, self.values)
        self.assertTrue(torch.allclose(align.rewards, self.rewards.to(align.device)))
        self.assertTrue(torch.allclose(align.c, torch.tensor(self.c_list, device=align.device)))
    
    def test_optimize_lambda(self):
        """Test lambda optimization."""
        align = AlignValues(
            value_list=self.values,
            rewards=self.rewards,
            c_list=self.c_list
        )
        
        # Optimize lambda values
        lambda_values, success = align.optimize_lambda(
            max_steps=100,
            verbose=False
        )
        
        # Check results
        self.assertIsInstance(lambda_values, list)
        self.assertEqual(len(lambda_values), len(self.c_list))
        self.assertIsInstance(success, bool)
        
        # All lambda values should be positive
        for lv in lambda_values:
            self.assertGreaterEqual(lv, 0)
            
    def test_sequential_optimize_lambda(self):
        """Test sequential lambda optimization."""
        align = AlignValues(
            value_list=self.values,
            rewards=self.rewards,
            c_list=self.c_list
        )
        
        # Optimize lambda values sequentially
        lambda_values = align.sequential_optimize_lambda(
            max_steps=50,
            verbose=False,
            rounds=2
        )
        
        # Check results
        self.assertIsInstance(lambda_values, list)
        self.assertEqual(len(lambda_values), len(self.c_list))
        
        # All lambda values should be positive
        for lv in lambda_values:
            self.assertGreaterEqual(lv, 0)
    
    def test_align_values_function(self):
        """Test the align_values function."""
        lambda_values, success = align_values(
            values=self.values,
            rewards=self.rewards,
            target_palette=self.c_list,
            verbose=False
        )
        
        # Check results
        self.assertIsInstance(lambda_values, list)
        self.assertEqual(len(lambda_values), len(self.c_list))
        self.assertIsInstance(success, bool)
        
        # All lambda values should be positive
        for lv in lambda_values:
            self.assertGreaterEqual(lv, 0)

if __name__ == "__main__":
    unittest.main() 