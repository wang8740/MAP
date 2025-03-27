"""Unit tests for language model adapters."""

import unittest
import torch
from unittest.mock import MagicMock, patch
from alignmap.models.language_models.adapter import LanguageModelAdapter

class TestLanguageModelAdapter(unittest.TestCase):
    """Tests for the LanguageModelAdapter class."""
    
    @patch('alignmap.models.language_models.adapter.AutoModelForCausalLM')
    @patch('alignmap.models.language_models.adapter.AutoTokenizer')
    def setUp(self, mock_tokenizer, mock_model):
        """Set up test fixtures with mocked transformers components."""
        # Configure mocks
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.eos_token = "<eos>"
        self.mock_tokenizer.batch_decode.return_value = ["Generated text 1", "Generated text 2"]
        mock_tokenizer.from_pretrained.return_value = self.mock_tokenizer
        
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mock_model.from_pretrained.return_value = self.mock_model
        
        # Create mock embedding
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.return_value = torch.ones((2, 3, 768))  # Batch size 2, seq len 3, dim 768
        self.mock_model.get_input_embeddings.return_value = self.mock_embeddings
        
        # Create adapter instance
        self.adapter = LanguageModelAdapter(
            model_name_or_path="gpt2",
            tokenizer=self.mock_tokenizer,
            model=self.mock_model
        )
    
    def test_initialization(self):
        """Test initialization of LanguageModelAdapter."""
        self.assertEqual(self.adapter.model_name, "gpt2")
        self.assertEqual(self.adapter.tokenizer, self.mock_tokenizer)
        self.assertEqual(self.adapter.model, self.mock_model)
        self.assertEqual(self.adapter.tokenizer.pad_token, "<eos>")
    
    def test_generate(self):
        """Test text generation."""
        # Single prompt
        result = self.adapter.generate(
            prompts="Hello, world!",
            max_length=10,
            temperature=0.7
        )
        
        # Check result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Generated text 1")
        
        # Check that tokenizer and model were called correctly
        self.mock_tokenizer.assert_called_once()
        self.mock_model.generate.assert_called_once()
        
        # List of prompts
        self.mock_tokenizer.reset_mock()
        self.mock_model.generate.reset_mock()
        
        result = self.adapter.generate(
            prompts=["Hello, world!", "How are you?"],
            max_length=10,
            temperature=0.7
        )
        
        # Check result
        self.assertEqual(len(result), 2)
        self.assertEqual(result, ["Generated text 1", "Generated text 2"])
    
    def test_get_embedding(self):
        """Test getting embeddings for text."""
        # Single text
        embeddings = self.adapter.get_embedding("Hello, world!")
        
        # Check embeddings shape
        self.assertEqual(embeddings.shape[0], 1)  # Batch size 1
        self.assertEqual(embeddings.dim(), 2)     # Should be 2D after mean pooling
        
        # Ensure model methods were called correctly
        self.mock_tokenizer.assert_called_once()
        self.mock_model.get_input_embeddings.assert_called_once()
        self.mock_embeddings.assert_called_once()
        
        # List of texts
        self.mock_tokenizer.reset_mock()
        self.mock_model.get_input_embeddings.reset_mock()
        self.mock_embeddings.reset_mock()
        
        embeddings = self.adapter.get_embedding(["Hello, world!", "How are you?"])
        
        # Check embeddings shape
        self.assertEqual(embeddings.shape[0], 2)  # Batch size 2
        self.assertEqual(embeddings.dim(), 2)     # Should be 2D after mean pooling
        
        # Ensure methods were called correctly
        self.mock_tokenizer.assert_called_once()
        self.mock_model.get_input_embeddings.assert_called_once()
        self.mock_embeddings.assert_called_once()

if __name__ == "__main__":
    unittest.main() 