# import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire
import torch


def soup(model_path1, model_path2, harmless_lambda, save_path):
    """
    Apply linear interpolation to two models to obtain and save a set of models.

    This function performs model interpolation between two pre-trained models
    based on a given interpolation factor (harmless_lambda).

    Args:
        model_path1 (str): Path to the first pre-trained model.
        model_path2 (str): Path to the second pre-trained model.
        harmless_lambda (float): Interpolation factor in [0, 1]. 
                                 0 means full weight to model1, 1 means full weight to model2.
        save_path (str): Path to save the interpolated model.

    Example:
        >>> model_path1 = "modelsDPO/basemodel-1000sample-0.1beta-0.0harmless"
        >>> model_path2 = "modelsDPO/basemodel-1000sample-0.1beta-1.0harmless"
        >>> harmless_lambda = 0.5
        >>> save_path = "soupModel/interpolated_model"
        >>> soup(model_path1, model_path2, harmless_lambda, save_path)

    Command-line usage:
        >>> python getDPOsoup.py --model_path1=model_path1 --model_path2=model_path2 --harmless_lambda=0.5 --save_path=soupModel_relative_path
    """

    # Load the models for interpolation
    model1 = AutoModelForCausalLM.from_pretrained(model_path1)
    model2 = AutoModelForCausalLM.from_pretrained(model_path2)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path1)
    
    # Perform model interpolation
    new_model_weights = {}
    with torch.no_grad():  # Use torch.no_grad() for efficient weight interpolation
        for key in model1.state_dict().keys():
            new_model_weights[key] = (1 - harmless_lambda) * model1.state_dict()[key] + harmless_lambda * model2.state_dict()[key]
    
    # Create a new model instance for saving
    model1.load_state_dict(new_model_weights)
    
    save_model_and_tokenizer(model1, tokenizer, save_path)


def save_model_and_tokenizer(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    fire.Fire(soup)