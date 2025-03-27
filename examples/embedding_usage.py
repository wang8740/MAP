#!/usr/bin/env python
"""
Example demonstrating how to use the embedding functionality in LanguageModelAdapter.

This script shows:
1. Loading a language model via the adapter
2. Converting text to embeddings
3. Computing semantic similarity between texts
4. Clustering similar texts
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

from alignmap.models.language_models.adapter import LanguageModelAdapter
from alignmap.utils.device import get_device

def get_embeddings(texts: List[str], model_name: str = "gpt2", device=None):
    """Get embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        model_name: Name or path of the language model
        device: Device to run the model on
        
    Returns:
        torch.Tensor: Embeddings for each text
    """
    print(f"Getting embeddings for {len(texts)} texts using {model_name}...")
    
    try:
        # Load model
        model = LanguageModelAdapter(model_name, device=device)
        
        # Get embeddings
        embeddings = model.get_embedding(texts)
        
        print(f"  - Embedding shape: {embeddings.shape}")
        return embeddings
    
    except Exception as e:
        print(f"  - Could not load language model: {e}")
        print("  - Creating mock embeddings for demonstration")
        
        # Create mock embeddings
        mock_embeddings = torch.randn(len(texts), 768)  # Pretend dimension is 768
        return mock_embeddings

def compute_similarities(embeddings: torch.Tensor):
    """Compute pairwise cosine similarities between text embeddings.
    
    Args:
        embeddings: Tensor of text embeddings
        
    Returns:
        numpy.ndarray: Similarity matrix
    """
    print("Computing pairwise similarities...")
    
    # Convert to numpy for sklearn
    embeddings_np = embeddings.cpu().numpy()
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(embeddings_np)
    
    return similarity_matrix

def visualize_embeddings(embeddings: torch.Tensor, texts: List[str], n_clusters: int = 3):
    """Visualize embeddings using PCA and K-means clustering.
    
    Args:
        embeddings: Tensor of text embeddings
        texts: List of texts corresponding to embeddings
        n_clusters: Number of clusters to form
    """
    print("Visualizing embeddings...")
    try:
        # Convert to numpy for sklearn
        embeddings_np = embeddings.cpu().numpy()
        
        # Reduce dimensions with PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_np)
        
        # Cluster with K-means
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings_np)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        
        # Plot points
        for i, cluster in enumerate(np.unique(clusters)):
            plt.scatter(
                embeddings_2d[clusters == cluster, 0],
                embeddings_2d[clusters == cluster, 1],
                label=f'Cluster {cluster}'
            )
        
        # Add text labels
        for i, (x, y) in enumerate(embeddings_2d):
            plt.annotate(
                f"Text {i+1}",
                (x, y),
                fontsize=8,
                alpha=0.7
            )
        
        plt.title('Text Embeddings Visualization (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save figure
        plt.savefig('embeddings_visualization.png')
        print("  - Visualization saved to 'embeddings_visualization.png'")
        
    except Exception as e:
        print(f"  - Could not visualize embeddings: {e}")

def main():
    """Run the embedding example."""
    print("Language Model Embedding Example")
    print("================================")
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Define some example texts
    texts = [
        "Neural networks are a class of machine learning models.",
        "Deep learning uses neural networks with many layers.",
        "Machine learning algorithms learn from data.",
        "Computer vision is a field of artificial intelligence.",
        "Natural language processing helps computers understand text.",
        "Reinforcement learning is about taking actions to maximize rewards.",
        "Supervised learning requires labeled data for training.",
        "Unsupervised learning finds patterns without labeled data.",
        "Transfer learning uses knowledge from one task for another.",
        "Generative AI can create new content like text and images."
    ]
    
    # Print texts
    print("\nExample texts:")
    for i, text in enumerate(texts):
        print(f"  Text {i+1}: {text}")
    
    # Get embeddings
    embeddings = get_embeddings(texts, model_name="gpt2", device=device)
    
    # Compute similarities
    similarity_matrix = compute_similarities(embeddings)
    
    # Print similarities
    print("\nSimilarity matrix (sample):")
    np.set_printoptions(precision=2, suppress=True)
    print(similarity_matrix[:5, :5])  # Print a subset for clarity
    
    # Find most similar pair
    if len(texts) > 1:  # Only if we have multiple texts
        similarity_mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(similarity_mask, False)  # Exclude self-similarity
        most_similar_idx = np.unravel_index(
            np.argmax(similarity_matrix * similarity_mask), 
            similarity_matrix.shape
        )
        
        print("\nMost similar pair of texts:")
        print(f"  Text {most_similar_idx[0]+1}: {texts[most_similar_idx[0]]}")
        print(f"  Text {most_similar_idx[1]+1}: {texts[most_similar_idx[1]]}")
        print(f"  Similarity: {similarity_matrix[most_similar_idx]:.4f}")
    
    # Visualize embeddings
    visualize_embeddings(embeddings, texts)
    
    print("\nExample completed!")

if __name__ == "__main__":
    main() 