"""Unit tests for Pareto optimization methods."""

import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from alignmap.core.alignment import AlignValues
from alignmap.core.pareto import (
    ParetoOptimizer,
    find_pareto_by_interpolation,
    find_pareto_by_grid_search,
    find_pareto_by_one_value
)

class TestParetoOptimizer(unittest.TestCase):
    """Tests for the ParetoOptimizer class."""
    
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
        
        # Create an AlignValues instance to use for testing
        self.aligner = AlignValues(
            value_list=self.values,
            rewards=self.rewards,
            c_list=self.c_list
        )
        
        # Mock the optimize_lambda method to always return success
        self.original_optimize_lambda = self.aligner.optimize_lambda
        self.aligner.optimize_lambda = MagicMock(return_value=([1.0, 0.5, 0.3], True))
        
        # Create the optimizer
        self.optimizer = ParetoOptimizer(self.aligner)
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore the original method
        self.aligner.optimize_lambda = self.original_optimize_lambda
    
    def test_initialization(self):
        """Test initialization of ParetoOptimizer."""
        self.assertEqual(self.optimizer.aligner, self.aligner)
    
    def test_find_pareto_by_interpolation(self):
        """Test finding a Pareto frontier point by interpolation."""
        c_low = [0.1, 0.1, 0.1]
        c_high = [0.9, 0.9, 0.9]
        
        # Test successful case
        success_rate, best_c, best_lambda = self.optimizer.find_pareto_by_interpolation(c_low, c_high)
        
        # Check results
        self.assertIsInstance(success_rate, float)
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)
        self.assertIsInstance(best_c, list)
        self.assertEqual(len(best_c), len(c_low))
        self.assertIsInstance(best_lambda, list)
        
        # Test failure case with mocked optimize_lambda
        self.aligner.optimize_lambda = MagicMock(return_value=([0.0, 0.0, 0.0], False))
        success_rate, best_c, best_lambda = self.optimizer.find_pareto_by_interpolation(c_low, c_high)
        
        # Check results
        self.assertEqual(success_rate, 0.0)
        self.assertIsNone(best_c)
        self.assertIsNone(best_lambda)
    
    def test_find_pareto_by_grid_search(self):
        """Test finding a Pareto frontier point by grid search."""
        c_low = [0.1, 0.1, 0.1]
        c_high = [0.9, 0.9, 0.9]
        grid_points = 5
        
        # Test successful case
        success_rate, best_c, best_lambda = self.optimizer.find_pareto_by_grid_search(
            c_low, c_high, grid_points
        )
        
        # Check results
        self.assertIsInstance(success_rate, float)
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)
        self.assertIsInstance(best_c, list)
        self.assertEqual(len(best_c), len(c_low))
        self.assertIsInstance(best_lambda, list)
        
        # Test failure case with mocked optimize_lambda
        self.aligner.optimize_lambda = MagicMock(return_value=([0.0, 0.0, 0.0], False))
        success_rate, best_c, best_lambda = self.optimizer.find_pareto_by_grid_search(
            c_low, c_high, grid_points
        )
        
        # Check results
        self.assertEqual(success_rate, 0.0)
        self.assertIsNone(best_c)
        self.assertIsNone(best_lambda)
    
    def test_find_pareto_by_one_value(self):
        """Test finding maximum feasible value for one constraint."""
        value_to_enhance = "helpfulness"
        
        # Set up mock rewards with a known maximum
        max_reward = 0.95
        dimension = self.values.index(value_to_enhance)
        self.aligner.rewards[dimension] = torch.tensor([0.5, 0.6, 0.7, 0.8, max_reward])
        
        # The binary search should converge to a value close to max_reward
        result = self.optimizer.find_pareto_by_one_value(value_to_enhance, precision=0.05)
        
        # Check result is a float and is close to max_reward
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, self.c_list[dimension])
        self.assertLessEqual(result, max_reward)
        
        # Test with invalid value name
        with self.assertRaises(ValueError):
            self.optimizer.find_pareto_by_one_value("invalid_value")

class TestParetoConvenienceFunctions(unittest.TestCase):
    """Tests for the Pareto optimization convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock AlignValues instance
        self.aligner = MagicMock()
        
        # Mock ParetoOptimizer methods
        self.aligner.device = "cpu"
        self.mock_pareto = MagicMock()
        self.mock_pareto.find_pareto_by_interpolation.return_value = (0.5, [0.5, 0.3], [1.0, 0.5])
        self.mock_pareto.find_pareto_by_grid_search.return_value = (0.6, [0.6, 0.4], [1.2, 0.6])
        self.mock_pareto.find_pareto_by_one_value.return_value = 0.7
        
        # Patch the ParetoOptimizer constructor
        self.patcher = patch('alignmap.core.pareto.ParetoOptimizer', return_value=self.mock_pareto)
        self.MockParetoOptimizer = self.patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    def test_find_pareto_by_interpolation_function(self):
        """Test the find_pareto_by_interpolation convenience function."""
        c_low = [0.1, 0.1]
        c_high = [0.9, 0.9]
        
        # Call the function
        result = find_pareto_by_interpolation(self.aligner, c_low, c_high)
        
        # Check the ParetoOptimizer was created with the aligner
        self.MockParetoOptimizer.assert_called_once_with(self.aligner)
        
        # Check the method was called with the right arguments
        self.mock_pareto.find_pareto_by_interpolation.assert_called_once_with(c_low, c_high)
        
        # Check the result is correct
        self.assertEqual(result, (0.5, [0.5, 0.3], [1.0, 0.5]))
    
    def test_find_pareto_by_grid_search_function(self):
        """Test the find_pareto_by_grid_search convenience function."""
        c_low = [0.1, 0.1]
        c_high = [0.9, 0.9]
        grid_points = 10
        
        # Call the function
        result = find_pareto_by_grid_search(self.aligner, c_low, c_high, grid_points)
        
        # Check the ParetoOptimizer was created with the aligner
        self.MockParetoOptimizer.assert_called_once_with(self.aligner)
        
        # Check the method was called with the right arguments
        self.mock_pareto.find_pareto_by_grid_search.assert_called_once_with(c_low, c_high, grid_points)
        
        # Check the result is correct
        self.assertEqual(result, (0.6, [0.6, 0.4], [1.2, 0.6]))
    
    def test_find_pareto_by_one_value_function(self):
        """Test the find_pareto_by_one_value convenience function."""
        value_to_enhance = "helpfulness"
        precision = 0.01
        
        # Call the function
        result = find_pareto_by_one_value(self.aligner, value_to_enhance, precision)
        
        # Check the ParetoOptimizer was created with the aligner
        self.MockParetoOptimizer.assert_called_once_with(self.aligner)
        
        # Check the method was called with the right arguments
        self.mock_pareto.find_pareto_by_one_value.assert_called_once_with(value_to_enhance, precision)
        
        # Check the result is correct
        self.assertEqual(result, 0.7)

if __name__ == "__main__":
    unittest.main() 