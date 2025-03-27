"""Functions for calculating various types of reward scores."""

import logging
import nltk
from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

try:
    # Try to initialize nltk requirements
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download nltk punkt: {e}")

def cal_humor_probabilities(
    sentences: List[str], 
    model: Any, 
    tokenizer: Any
) -> Tuple[List[float], List[float]]:
    """Calculate humor probabilities and raw scores.

    Args:
        sentences (List[str]): List of sentences to evaluate
        model: Model for humor classification
        tokenizer: Tokenizer for input processing

    Returns:
        Tuple[List[float], List[float]]: Humor probabilities and raw scores
    """
    # Tokenize the sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Assuming the second column corresponds to the humor class
    probabilities = probs[:, 1].tolist()

    # Extract raw scores (logits) for the humor class
    raw_scores = outputs.logits[:, 1].tolist()

    return probabilities, raw_scores


def cal_positive_sentiment(
    sentences: List[str], 
    model: Any, 
    tokenizer: Any
) -> Tuple[List[float], List[float]]:
    """Calculate positive sentiment probabilities and scores.

    Args:
        sentences (List[str]): List of sentences to evaluate
        model: Model for sentiment classification
        tokenizer: Tokenizer for input processing

    Returns:
        Tuple[List[float], List[float]]: Positive probabilities and scores
    """
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probabilities = probs[:, 1].tolist()

    positive_scores = outputs.logits[:, 1].tolist()

    return probabilities, positive_scores


def cal_gpt2_harmless_probabilities(
    prompts: List[str], 
    continuations: List[str], 
    model: Any, 
    tokenizer: Any
) -> Tuple[List[float], List[float]]:
    """Calculate probabilities and raw scores indicating the harmlessness of a response.

    Args:
        prompts (List[str]): List of initial prompt strings
        continuations (List[str]): List of continuation (response) strings
        model: Model for evaluating harmlessness
        tokenizer: Tokenizer for processing the inputs

    Returns:
        Tuple[List[float], List[float]]: Harmlessness probabilities and raw scores
    """
    # Format inputs for model
    questions = ["\n\nHuman: " + p + " \n\nAssistant:" for p in prompts]
    probabilities = []
    raw_scores = []

    with torch.no_grad():
        for q, a in zip(questions, continuations):
            inputs = tokenizer(q, a, return_tensors='pt', truncation=True).to(model.device)
            outputs = model(**inputs)

            prob = torch.sigmoid(outputs.logits).squeeze().item()
            probabilities.append(prob)

            raw_score = outputs.logits.squeeze().item()
            raw_scores.append(raw_score)

    return probabilities, raw_scores


def cal_gpt2_helpful_probabilities(
    prompts: List[str], 
    continuations: List[str], 
    model: Any, 
    tokenizer: Any
) -> Tuple[List[float], List[float]]:
    """Calculate probabilities and raw scores indicating the helpfulness of a response.

    Args:
        prompts (List[str]): List of prompt strings
        continuations (List[str]): List of continuation (response) strings
        model: Model for evaluating helpfulness
        tokenizer: Tokenizer for processing the inputs

    Returns:
        Tuple[List[float], List[float]]: Helpfulness probabilities and raw scores
    """
    questions = ["\n\nHuman: " + p + " \n\nAssistant:" for p in prompts]
    probabilities = []
    raw_scores = []
    
    with torch.no_grad():
        for q, a in zip(questions, continuations):
            inputs = tokenizer(q, a, return_tensors='pt', truncation=True).to(model.device)
            outputs = model(**inputs)

            prob = torch.sigmoid(outputs.logits).squeeze().item()
            probabilities.append(prob)

            raw_score = outputs.logits.squeeze().item()
            raw_scores.append(raw_score)

    return probabilities, raw_scores


def compute_rep_n(sentence: str, n: int) -> float:
    """Calculate the unique fraction of n-grams within a sentence.

    Args:
        sentence (str): Sentence to analyze
        n (int): Length of the n-grams to compute

    Returns:
        float: Fraction of unique n-grams in the sentence
    """
    try:
        tokens = nltk.word_tokenize(sentence)
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        
        if not ngrams:
            return 1.0  # Handle case with no n-grams
            
        uniq_n = len(set(ngrams)) / (len(ngrams) + 1)
        return uniq_n
    except Exception as e:
        logger.warning(f"Error computing n-grams: {e}")
        return 0.5  # Fallback value


def cal_diversity(sentence: str) -> float:
    """Calculate diversity score by computing unique n-grams.

    Args:
        sentence (str): Sentence to analyze

    Returns:
        float: Diversity score, product of unique n-grams for each n in (2, 3, 4)
    """
    diversity = 1.0
    for n in range(2, 5):
        rep_n_val = compute_rep_n(sentence, n)
        diversity *= rep_n_val
        
    return diversity


def cal_log_perplexity(
    sentences: List[str], 
    model: Any, 
    tokenizer: Any
) -> List[float]:
    """Calculate log perplexity for a list of sentences.

    Args:
        sentences (List[str]): List of sentences to evaluate
        model: Model to evaluate perplexity
        tokenizer: Tokenizer for processing the inputs

    Returns:
        List[float]: List of log perplexity values for each sentence
    """
    log_perplexities = []
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            logp = loss.item()  # Get the scalar value of loss
            log_perplexities.append(logp)
            
    return log_perplexities


def cal_coherence(
    prompts: List[str], 
    continuations: List[str], 
    model: Any, 
    tokenizer: Any
) -> List[float]:
    """Calculate coherence between prompts and continuations using cosine similarity.

    Args:
        prompts (List[str]): List of prompt sentences
        continuations (List[str]): List of continuation sentences
        model: Model for sentence embedding
        tokenizer: Tokenizer for processing the inputs

    Returns:
        List[float]: List of cosine similarity scores for each prompt-continuation pair
    """
    prompt_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    continuation_inputs = tokenizer(continuations, padding=True, truncation=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        prompt_embeddings = model(**prompt_inputs, output_hidden_states=True, return_dict=True).pooler_output
        continuation_embeddings = model(**continuation_inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    similarity = [1 - cosine(prompt_embeddings[i].cpu(), continuation_embeddings[i].cpu()) 
                  for i in range(len(prompt_embeddings))]
    
    return similarity 