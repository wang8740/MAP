# import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire
import torch


def soup(basemodel_name, sample_size, beta, harmless_lambda, save_path):
    '''
        beta: same as in trainDPO.py
        harmless_lambda: Interpolation factor in [0, 1]
    '''
    model_path1 = f"modelsDPO/{basemodel_name}-{sample_size}sample-{beta}beta-0.0harmless"
    model_path2 = f"modelsDPO/{basemodel_name}-{sample_size}sample-{beta}beta-1.0harmless"
    
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
    # Save the model
    model.save_pretrained(save_path)
    
    # Save the tokenizer
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    fire.Fire(soup)