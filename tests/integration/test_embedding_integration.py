"""Integration test for the embedding functionality."""

import unittest
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from alignmap.models.language_models.adapter import LanguageModelAdapter
from alignmap.utils.device import get_device

class TestEmbeddingIntegration(unittest.TestCase):
    """Integration test for embeddings functionality.
    
    This test verifies that embeddings can be used effectively for
    semantic similarity comparison, which was one of the primary
    use cases in the original codebase.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            # Try to load a small model for testing
            self.model = LanguageModelAdapter("distilgpt2")
            self.skip_test = False
        except Exception as e:
            # If model can't be loaded, mark to skip the actual tests
            print(f"Could not load model: {e}")
            self.skip_test = True
    
    def test_semantic_similarity(self):
        """Test that embeddings can be used for semantic similarity."""
        if self.skip_test:
            self.skipTest("Model could not be loaded")
            
        # Create pairs of semantically similar texts
        similar_pairs = [
            (
                "Neural networks are used in deep learning models.",
                "Deep learning employs neural networks for AI tasks."
            ),
            (
                "The cat sat on the mat in the living room.",
                "A feline was sitting on the floor mat."
            ),
            (
                "The stock market declined dramatically yesterday.",
                "Yesterday saw a significant drop in stocks and shares."
            )
        ]
        
        # Create pairs of semantically different texts
        different_pairs = [
            (
                "Neural networks are used in deep learning models.",
                "The cat sat on the mat in the living room."
            ),
            (
                "Deep learning employs neural networks for AI tasks.",
                "The stock market declined dramatically yesterday."
            ),
            (
                "A feline was sitting on the floor mat.",
                "Yesterday saw a significant drop in stocks and shares."
            )
        ]
        
        # Get embeddings for all texts
        all_texts = []
        for pair in similar_pairs + different_pairs:
            all_texts.extend(pair)
            
        embeddings = self.model.get_embedding(all_texts)
        
        # Compute similarities
        embeddings_np = embeddings.cpu().numpy()
        similarities = []
        
        # Compute similarities for similar pairs
        for i in range(len(similar_pairs)):
            idx1 = i * 2
            idx2 = i * 2 + 1
            similarity = cosine_similarity([embeddings_np[idx1]], [embeddings_np[idx2]])[0][0]
            similarities.append(("similar", similarity))
        
        # Compute similarities for different pairs
        for i in range(len(different_pairs)):
            idx1 = (len(similar_pairs) + i) * 2
            idx2 = (len(similar_pairs) + i) * 2 + 1
            similarity = cosine_similarity([embeddings_np[idx1]], [embeddings_np[idx2]])[0][0]
            similarities.append(("different", similarity))
        
        # Get average similarity for each group
        similar_scores = [s[1] for s in similarities if s[0] == "similar"]
        different_scores = [s[1] for s in similarities if s[0] == "different"]
        
        avg_similar = sum(similar_scores) / len(similar_scores)
        avg_different = sum(different_scores) / len(different_scores)
        
        # Similar pairs should have higher similarity on average
        self.assertGreater(avg_similar, avg_different)
    
    def test_batch_processing(self):
        """Test batch processing of embeddings."""
        if self.skip_test:
            self.skipTest("Model could not be loaded")
            
        # Create a list of texts
        texts = [
            "This is the first text.",
            "Here is another text for embedding.",
            "The third text is a bit longer than the others.",
            "Text number four is here.",
            "Finally, we have the fifth text in the batch."
        ]
        
        # Get embeddings
        embeddings = self.model.get_embedding(texts)
        
        # Check shape
        self.assertEqual(embeddings.shape[0], len(texts))
        self.assertTrue(embeddings.shape[1] > 0)  # Should have some embedding dimension
        
        # Check that all embeddings are different
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Embeddings should be different for different texts
                self.assertFalse(torch.allclose(embeddings[i], embeddings[j]))

if __name__ == "__main__":
    unittest.main() 