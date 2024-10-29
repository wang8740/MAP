import re
import os
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM
)
import numpy as np
import json
import nltk
# nltk.download('punkt_tab') # run one time only and figure out the path
nltk.data.path.append('/users/5/wang8740/nltk_data')


from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import subprocess
# from trl.core import LengthSampler
import random

TASK_NAME = 'conversation' # 'conversation', 'sentiment_control'
if TASK_NAME == 'conversation':
    ALL_SUPPORTED_VALUES = ["humor", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"]
    ALL_SUPPORTED_VALUES_plotnames = ["Humor", "Helpfulness", "Harmlessness", "Diversity", "Coherence", "Perplexity"]
elif TASK_NAME == 'sentiment_control':
    ALL_SUPPORTED_VALUES = ["positive", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"]
    ALL_SUPPORTED_VALUES_plotnames = ["Positiveness", "Helpfulness", "Harmlessness", "Diversity", "Coherence", "Perplexity"]


def get_device() -> str:
    """Determines the device for model computation.

    Returns:
        str: Device type, either "cuda" if available, else "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_nvidia_smi_info() -> Union[str, None]:
    """Fetches NVIDIA GPU details using the nvidia-smi command.

    Returns:
        str or None: Output from nvidia-smi command if successful, None otherwise.
    """
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to run nvidia-smi: {e}")

devices = {
    "llama2_chat": "cuda",  # Use "balanced" if you want to perform model parallel using accelerate
    "humor": get_device(),
    "positive": get_device(),
    # "harmless": get_device(),
    "gpt2-helpful": get_device(),
    "gpt2-harmless": get_device(),
    "diversity": get_device(),
    "coherence": get_device(),
    "perplexity": get_device(),
    "gpt2": get_device(),
    "opt1.3b": get_device(),
}

def save_results_to_json(results: list, file_path: str) -> None:
    """Saves results to a JSON file.

    Args:
        results (list): List of dictionaries containing prompt and generated text.
        file_path (str): Path to save the JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {file_path}")


def clean_and_trim_to_last_sentence(prompts: List[str], decoded_outputs: List[str]) -> List[str]:
    """Cleans and trims decoded outputs to the last complete sentence.

    Args:
        prompts (List[str]): List of input prompts.
        decoded_outputs (List[str]): List of decoded model outputs.

    Returns:
        List[str]: Cleaned and trimmed outputs.
    """
    clean_outputs = []
    for prompt, s in zip(prompts, decoded_outputs):
        
        clean_s = s.replace('\n', ' ').strip()

        # Replace comma at the end with a period
        if clean_s.endswith(','):
            clean_s = clean_s[:-1] + '.'

        # This regex attempts to find the last complete sentence by looking for sentence terminators like 
        # periods, question marks, or exclamation marks followed by whitespace or end of string.
        text = re.findall(r'.*?[.!?](?=\s|$)', clean_s)
        trim_s = ' '.join(text).strip() if text else prompt

        clean_outputs.append(trim_s)
        
    return clean_outputs


class LengthSampler:
    """A class for sampling text lengths within a given range."""

    def __init__(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self):
        """Generates a random length within the specified range.

        Returns:
            int: Random length within the min and max range.
        """
        return random.randint(self.min_length, self.max_length)

    def trim_to_word_boundary(self, text, length):
        """Trims text to the last word boundary within a given length.

        Args:
            text (str): Text to be trimmed.
            length (int): Maximum character length for trimming.

        Returns:
            str: Text trimmed to the last word boundary.
        """
        if len(text) <= length:
            return text
        # Find the last space within the length
        end = text[:length].rfind(' ')
        if end == -1:
            # If no space is found, return the full text up to the length
            return text[:length]
        return text[:end]


def get_prompts_from_Anthropic_harmless():
    """Retrieves harmless prompts from the Anthropic dataset.
    Importantly, the generation is non-random everytime being called.

    Returns:
        List[str]: List of harmless prompts.
    """

    # Load the 'test' partition of the 'harmless-base' subset
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split='test')

    # Initialize a list to store the prompts
    prompts = []

    # Extract the prompts from the 'chosen' entries
    for item in dataset:
        chosen_text = item['chosen']
        # Extract the text after "Human:" and before "Assistant:"
        start_idx = chosen_text.find("Human: ") + len("Human: ")
        end_idx = chosen_text.find("Assistant:")
        if start_idx != -1 and end_idx != -1:
            prompt = chosen_text[start_idx:end_idx].strip()
            prompts.append(prompt)

    # You can now use the list `prompts` for further processing
    print(f"get_prompts_from_Anthropic_harmless: {len(prompts)} total prompts")
    print("\n10 example prompts:\n", prompts[:10])  # Print the first 10 prompts to check
    return prompts


def get_prompts_from_imdb():
    """Retrieves prompts from the IMDB dataset, filtered by length.
    Importantly, the generation is non-random everytime being called.

    Returns:
        List[str]: List of filtered IMDB prompts.
    """

    # Load the IMDB dataset with the 'datasets' library
    ds = load_dataset("imdb", split="train")
        
    # Filter to keep only reviews longer than 20 characters
    ds = ds.filter(lambda x: len(x["text"]) > 30, batched=False) 
    # ds = ds.filter(lambda x: len(x["text"]) <= 500, batched=False)
    
    # Reduce the sample size for a quick demo
    ds = ds.select(range(min(1000, len(ds)))) #1000 total prompts
    
    # Initialize a list to store the prompts
    input_size = LengthSampler(20, 30)
    prompts = [input_size.trim_to_word_boundary(x['text'], input_size()) for x in ds]

    print(f"get_prompts_from_imdb: {len(prompts)} total prompts")
    print("\n10 example prompts:\n", prompts[:10])  # Print the first 10 prompts to check

    return prompts


def cal_humor_probabilities(sentences, model, tokenizer):
    """Calculates humor probabilities and raw scores.

    Args:
        sentences (List[str]): List of sentences, whose size/batchsize is limited by memory
        model: Model for humor classification.
        tokenizer: Tokenizer for input processing.

    Returns:
        Tuple[List[float], List[float]]: Humor probabilities and raw scores.
    """

    # Tokenize the sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(devices["humor"])

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Assuming the second column corresponds to the humor class
    probabilities = probs[:, 1].tolist()

    # Extract raw scores (logits) for the humor class
    raw_scores = outputs.logits[:, 1].tolist()

    return probabilities, raw_scores


def cal_positive_sentiment(sentences, model, tokenizer):
    """Calculates positive sentiment probabilities and scores.

    Args:
        sentences (List[str]): List of sentences.
        model: Model for sentiment classification.
        tokenizer: Tokenizer for input processing.

    Returns:
        Tuple[List[float], List[float]]: Positive probabilities and scores.
    """
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(devices["positive"])

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probabilities = probs[:, 1].tolist()

    # negative_scores = outputs.logits[:, 0].tolist()
    positive_scores = outputs.logits[:, 1].tolist()

    return probabilities, positive_scores

def cal_harmless_probabilities(sentences: List[str], model, tokenizer) -> Tuple[List[float], List[float]]:
    """Calculates harmlessness probabilities and scores.

    Args:
        sentences (List[str]): List of sentences.
        model: Model for harmlessness classification.
        tokenizer: Tokenizer for input processing.

    Returns:
        Tuple[List[float], List[float]]: Harmless probabilities and scores.
    """

    # Tokenize the sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(devices["harmless"])

    with torch.no_grad():
        outputs = model(**inputs)

    # Do not use score. Use probability for better interpretability
#     scores = outputs.logits[:, 0].tolist()

    probabilities = torch.sigmoid(outputs.logits).squeeze().tolist()

    # Extract raw scores (logits)
    raw_scores = outputs.logits.squeeze().tolist()

    return probabilities, raw_scores


def cal_gpt2_harmless_probabilities(prompts: List[str], continuations: List[str], model, tokenizer) -> Tuple[List[float], List[float]]:
    """Calculates probabilities and raw scores indicating the harmlessness of a response.

    Args:
        prompts (List[str]): List of initial prompt strings.
        continuations (List[str]): List of continuation (response) strings.
        model: Model for evaluating harmlessness.
        tokenizer: Tokenizer for processing the inputs.

    Returns:
        Tuple[List[float], List[float]]: Harmlessness probabilities and raw scores.
    """
    # arrange model inputs in model-recognizable format
    questions = ["\n\nHuman: "+p+" \n\nAssistant:" for p in prompts]
    answers = continuations
    probabilities = []
    raw_scores = []

    with torch.no_grad():
        for q, a in zip(questions, answers):
            inputs = tokenizer(q, a, return_tensors='pt', truncation=True).to(devices["gpt2-harmless"])
            outputs = model(**inputs)

            prob = torch.sigmoid(outputs.logits).squeeze().item()
            probabilities.append(prob)

            raw_score = outputs.logits.squeeze().item()
            raw_scores.append(raw_score)

    return probabilities, raw_scores


def cal_gpt2_helpful_probabilities(prompts: List[str], continuations: List[str], model, tokenizer) -> Tuple[List[float], List[float]]:
    """Calculates probabilities and raw scores indicating the helpfulness of a response.

    Args:
        prompts (List[str]): List of prompt strings.
        continuations (List[str]): List of continuation (response) strings.
        model: Model for evaluating helpfulness.
        tokenizer: Tokenizer for processing the inputs.

    Returns:
        Tuple[List[float], List[float]]: Helpfulness probabilities and raw scores.
    """
    questions = ["\n\nHuman: "+p+" \n\nAssistant:" for p in prompts]
    answers = continuations
    probabilities = []
    raw_scores = []
    with torch.no_grad():
        for q, a in zip(questions, answers):
            inputs = tokenizer(q, a, return_tensors='pt', truncation=True).to(devices["gpt2-helpful"])
            outputs = model(**inputs)

            prob = torch.sigmoid(outputs.logits).squeeze().item()
            probabilities.append(prob)

            raw_score = outputs.logits.squeeze().item()
            raw_scores.append(raw_score)

    return probabilities, raw_scores


def compute_rep_n(sentence: str, n: int) -> float:
    """Calculates the unique fraction of n-grams within a sentence.
    This function is a subroutine of cal_diversity

    Args:
        sentence (str): Sentence to analyze.
        n (int): Length of the n-grams to compute.

    Returns:
        float: Fraction of unique n-grams in the sentence.
    """
    tokens = word_tokenize(sentence)
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    uniq_n = len(set(ngrams)) / (len(ngrams) + 1)
    
    return uniq_n


def cal_diversity(sentence: str) -> float:
    """Calculates diversity score by computing unique n-grams for n=2,3,4.

    Args:
        sentence (str): Sentence to analyze.

    Returns:
        float: Diversity score, product of unique n-grams for each n in (2, 3, 4).
    """
    
    diversity = 1.0
    for n in range(2, 5):
        rep_n_val = compute_rep_n(sentence, n)
        diversity *= rep_n_val
        
    return diversity


def cal_log_perplexity(sentences: List[str], model, tokenizer) -> List[float]:
    """Calculates log perplexity for a list of sentences.

    Args:
        sentences (List[str]): List of sentences.
        model: Model to evaluate perplexity.
        tokenizer: Tokenizer for processing the inputs.

    Returns:
        List[float]: List of log perplexity values for each sentence.
    """
    log_perplexities = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(devices["perplexity"])
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            logp = loss.item()  # Get the scalar value of loss
            log_perplexities.append(logp)
            
    return log_perplexities


def cal_coherence(prompts: List[str], continuations: List[str], model, tokenizer) -> List[float]:
    """Calculates coherence between prompts and continuations using cosine similarity.

    Args:
        prompts (List[str]): List of prompt sentences.
        continuations (List[str]): List of continuation sentences.  Equal length as prompts as each prompt corresponds to each sentence.
        model: Model for sentence embedding.
        tokenizer: Tokenizer for processing the inputs.

    Returns:
        List[float]: List of cosine similarity scores for each prompt-continuation pair.
    """
    
    prompt_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(devices["coherence"])
    continuation_inputs = tokenizer(continuations, padding=True, truncation=True, return_tensors="pt").to(devices["coherence"])
    with torch.no_grad():
        prompt_embeddings = model(**prompt_inputs, output_hidden_states=True, return_dict=True).pooler_output
        continuation_embeddings = model(**continuation_inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    similarity = [1 - cosine(prompt_embeddings[i].cpu(), continuation_embeddings[i].cpu()) for i in range(len(prompt_embeddings))]
    
    return similarity


def get_reward(sentences: List[str], value: str, model=None, tokenizer=None, prompts: List[str] = None, use_score: bool = True) -> Union[List[float], None]:
    """Calculates reward scores based on the specified value (e.g., humor, coherence).

    Args:
        sentences (List[str]): List of sentences for evaluation. Each consists of a prompt and a continuation.
        value (str): Type of reward (e.g., "humor", "positive", etc.).
        model: Model used for evaluation, if applicable.
        tokenizer: Tokenizer used for model input processing.
        prompts (List[str], optional): List of prompts if needed for evaluation. Required to compute the gpt2_harmless, gpt2_helpful, coherence.
        use_score (bool, optional): If True, returns raw scores; otherwise, probabilities.

    Returns:
        Union[List[float], None]: List of reward scores or probabilities, or None if the value is invalid.

    Example:
        >>> rewards = get_reward(sentences, "humor")
        >>> print("Average Reward:", np.mean(rewards))
    """
    
    if value == "humor":
        humor_prob, humor_score = cal_humor_probabilities(sentences, model, tokenizer)
        if use_score:
            rewards = humor_score
        else:
            # rewards = [np.log(p) for p in humor_prob]
            rewards = humor_prob

    elif value == "positive":
        positive_prob, positive_scores = cal_positive_sentiment(sentences, model, tokenizer)
        if use_score:
            rewards = positive_scores
        else:
            # rewards = [np.log(p) for p in positive_prob]
            rewards = positive_prob

    elif value == "harmless":
        harmless_prob, harmless_score = cal_harmless_probabilities(sentences, model, tokenizer)
        if use_score:
            rewards = harmless_score
        else:
            rewards = [np.log(p) for p in harmless_prob]

    elif value == "gpt2-harmless":
        continuations = [s[len(p):].strip() for s, p in zip(sentences, prompts)]
        harmless_prob, harmless_score = cal_gpt2_harmless_probabilities(prompts, continuations, model, tokenizer)
        if use_score:
            rewards = harmless_score
        else:
            rewards = [np.log(p) for p in harmless_prob]
        
    elif value == "gpt2-helpful":
        continuations = [s[len(p):].strip() for s, p in zip(sentences, prompts)]
        helpful_prob, helpful_score = cal_gpt2_helpful_probabilities(prompts, continuations, model, tokenizer)
        if use_score:
            rewards = helpful_score
        else:
            rewards = [np.log(p) for p in helpful_prob]
        
    elif value == "diversity":
        rewards = [cal_diversity(s) for s in sentences]
        
    elif value == "coherence":
        # Assume `prompts` and `sentences` are lists of equal length where each prompt corresponds to each sentence
        # Stripping prompt part from sentence
        continuations = [s[len(p):].strip() for s, p in zip(sentences, prompts)]
        rewards = cal_coherence(prompts, continuations, model, tokenizer)

    elif value == "perplexity":
        # Use negative log perplexity as a reward, assuming lower perplexity is better
        # Note that the model and tokenizer here should be generative model and tokenizer
        log_perplexity = cal_log_perplexity(sentences, model, tokenizer)
        rewards = [-logp for logp in log_perplexity]  

    else:
        rewards = None
        
    return rewards


def convert_ppo_modelname_to_huggingface_valid(ppo_model_name: str) -> str:
    """Converts a PPO-trained model name to a valid Hugging Face model path format.

    Args:
        ppo_model_name (str): Model name to be converted.

    Returns:
        str: Converted model name.
    """
    return ppo_model_name.replace(',', '_').replace('=', '_')


def get_model_and_tokenizer(model_name: str):
    """Fetches a model and tokenizer based on the model name.

    Args:
        model_name (str): Name of the model.

    Returns:
        Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]: The specified model and tokenizer.
    """
    # Default to 'cuda' if model name is not in the devices dictionary
    device = devices.get(model_name, "cuda")

    # Support both exact matches and substring matches for model names, since some will be PPO finetuned/derived from the original model

    if "llama2_chat" in model_name:
        # Assuming you want to support both exact matches and substring matches for model names
        model_path = "/home/aanwar/wang8740/llama/model_output/llama-2-7b-chat" if "llama2_chat" == model_name else model_name
        model = LlamaForCausalLM.from_pretrained(model_path, device_map=device)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

    elif 'opt1.3b' in model_name:
        # ref: https://huggingface.co/blog/introducing-csearch#62-example-two---opt
        opt_path = "facebook/opt-1.3b" if 'opt1.3b' == model_name else model_name
        model = OPTForCausalLM.from_pretrained(opt_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(opt_path, padding_side='left')

    elif 'humor' in model_name:
        humor_path = "mohameddhiab/humor-no-humor" if 'humor' == model_name else model_name
        model = AutoModelForSequenceClassification.from_pretrained(humor_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(humor_path)

    elif 'positive' in model_name:
        positive_path = "lvwerra/distilbert-imdb" if 'positive' == model_name else model_name
        model = AutoModelForSequenceClassification.from_pretrained(positive_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(positive_path)

    # elif 'harmless' in model_name:
    #     harmless_path = "OpenAssistant/reward-model-deberta-v3-large-v2" if 'harmless' == model_name else model_name
    #     model = AutoModelForSequenceClassification.from_pretrained(harmless_path, device_map=device)
    #     tokenizer = AutoTokenizer.from_pretrained(harmless_path)

    elif 'gpt2-harmless' in model_name:
        harmless_path = "Ray2333/gpt2-large-harmless-reward_model" if 'gpt2-harmless' == model_name else model_name
        model = AutoModelForSequenceClassification.from_pretrained(harmless_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(harmless_path)

    elif "gpt2-helpful" in model_name:
        model_path = "Ray2333/gpt2-large-helpful-reward_model" if 'gpt2-helpful' == model_name else model_name
        model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    elif 'coherence' in model_name:
        # SimCSE sentence embedding model from https://github.com/princeton-nlp/SimCSE
        SimCSE_path = "princeton-nlp/sup-simcse-bert-base-uncased" if 'coherence' == model_name else model_name
        model = AutoModel.from_pretrained(SimCSE_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(SimCSE_path)

    # This must be placed after 'gpt2-harmless', 'gpt2-helpful', etc. to avoid confusion
    elif 'gpt2' in model_name:
        # ref: https://huggingface.co/blog/introducing-csearch#41-deterministic-methods
        model_path = "gpt2" if 'gpt2' == model_name else model_name
        model = GPT2LMHeadModel.from_pretrained(model_path, device_map=device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

    else:
        raise ValueError(f"Invalid model_name: {model_name}")

    print(f"\nAttempting to load model {model_name}, device set to {device}")

    return model, tokenizer


if __name__ == "__main__":

    '''
        gen original prompt-continuation data
    '''
    get_prompts_from_imdb()