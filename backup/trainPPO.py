import torch
from tqdm import tqdm
import fire
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import Dataset
from utils import (
    get_prompts_from_Anthropic_harmless,
    get_prompts_from_imdb,
    get_model_and_tokenizer,
    get_reward,
    ALL_SUPPORTED_VALUES,
    convert_ppo_modelname_to_huggingface_valid,
    get_nvidia_smi_info
)
from torch.nn.utils.rnn import pad_sequence
import os

def collator(data):
    """Data collator function for grouping data batches without padding. 
    PPOTrainer handles padding internally based on the tokenizer settings.

    Args:
        data (list): List of data samples.

    Returns:
        dict: A dictionary with collated data grouped by each key in the input.
    """
    # PPOTrainer will handle padding using tokenizer
    # https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L488 (line 566)
    return dict((key, [d[key] for d in data]) for key in data[0])

def build_dataset(config, tokenizer, data_name):
    """Builds and tokenizes the dataset for training based on specified data name.

    Args:
        config (PPOConfig): Configuration for PPO training.
        tokenizer (AutoTokenizer): Tokenizer used to process and encode text data.
        data_name (str): Name of the dataset, either "Imdb" or "Anthropic-harmless".

    Returns:
        Dataset: A Hugging Face Dataset object with tokenized prompts for training.
    """
    # Load prompts using the specialized function
    if data_name=="Imdb":
        prompts = get_prompts_from_imdb()
    elif data_name=="Anthropic-harmless":
        prompts = get_prompts_from_Anthropic_harmless()
    else:
        raise ValueError(f"Data name {data_name} is not supported.")

    # Convert list of prompts into a Dataset object
    ds = Dataset.from_dict({'text': prompts})

    def tokenize(sample):
        input_ids = tokenizer.encode(sample['text'], add_special_tokens=True)
        input_ids = tokenizer.encode(sample['text'])
        query = tokenizer.decode(input_ids)
        return {'input_ids': input_ids, "query": query}

    # Apply tokenization map to the dataset
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type='torch')  # Set format for PyTorch

    return ds


def main(lam_list, value_list, model_name, data_name, save_path, learning_rate=1e-6, batch_size=20, mini_batch_size=2, nepoch=1):
    """Main function to train a model with PPO (Proximal Policy Optimization) based on user-defined parameters.

    Args:
        lam_list (list of float): List of lambda values for aligning specified values.
        value_list (str): Comma-separated string of values to align (or "all" for all values).
        model_name (str): Name of the model to use (e.g., "opt1.3b").
        data_name (str): Name of the dataset to use ("Imdb" or "Anthropic-harmless").
        save_path (str): Path to save the trained model.
        learning_rate (float): Learning rate for PPO training. Defaults to 1e-6.
        batch_size (int): Total batch size for training. Defaults to 20.
        mini_batch_size (int): Batch size for each step. Defaults to 2.
        nepoch (int): Number of training epochs. Defaults to 1.

    # Example command-line usage:
        >>> python trainPPO.py --model_name="opt-1.3b" --data_name="Imdb" --value_list="all" --lam_list="0.241,0.077,0.117,0.033,0.070,0.065" --learning_rate=1e-4

    """

    if model_name=="opt1.3b": 
        model_path_name = "facebook/opt-1.3b" 
    else:
        model_path_name = "/home/aanwar/wang8740/llama/model_output/llama-2-7b-chat"

    # check if batch_size is multiple of mini_batch_size
    if batch_size % mini_batch_size != 0:
        raise ValueError(
            'mini_batch_size * gradient_accumulation_steps should equal to batch size' 
        )
    
    config = PPOConfig(
        model_name=model_path_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=int(batch_size // mini_batch_size)
    )

    # Initialize tokenizer with padding token
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if 'gpt' in model_name or 'opt' in model_name or 'llama' in model_name:
        tokenizer.padding_side = 'left'

    dataset = build_dataset(config, tokenizer, data_name)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    if not isinstance(lam_list, (list, tuple)):
        lam_list = [lam_list]
    elif not isinstance(lam_list, list):
        lam_list = list(lam_list)
    else:
        lam_list = lam_list
    lam = torch.tensor(lam_list, dtype=torch.float32)
    lam_str = ','.join(f"{l:.3f}" for l in lam_list)

    values = ALL_SUPPORTED_VALUES if value_list == "all" else value_list.split(',')
    if value_list == "all":
        values_str = "all"
    else:
        values_str = ','.join(values)

    model_rewards, tokenizer_rewards = {}, {}
    feasible_values = [v for v in values if v not in ["diversity", "perplexity"]]
    feasible_values_indices = [i for i in range(len(values)) if values[i] not in ["diversity", "perplexity"]]
    for value in feasible_values:
        try:
            model_rewards[value], tokenizer_rewards[value] = get_model_and_tokenizer(value)
        except RuntimeError as e:
            print(f"Failed to load model or tokenizer for value {value}: {e}")
            continue

    print(f"\nInput lambda is {lam_str} for aligning {values_str} for PPO tuning")

    generation_config = {
        "max_new_tokens": 50,
        "temperature": 1,
        "top_k": 50,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch in tqdm(range(nepoch), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            # print("Batch received from DataLoader:", batch)
            # print("Type of input_ids:", type(batch['input_ids']))
            # print("Shape of input_ids:", batch['input_ids'].shape)            

            query_tensors = batch["input_ids"]
            # print("Query Tensors:", query_tensors)

            # having bos_token_id equal to eos_token_id causes a bug in PPOTrainer(it will only preserve the first eos token from output)
            # need to set remove_padding to False
            remove_padding = tokenizer.bos_token_id != tokenizer.eos_token_id
            response_tensors = ppo_trainer.generate(query_tensors, remove_padding=remove_padding, **generation_config)
            if not remove_padding:
                for i in range(len(response_tensors)):
                    pad_mask = response_tensors[i] == tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)
                    if pad_start.shape[0] != 1:
                        pad_start = pad_start[1, 0].item()
                        response_tensors[i] = response_tensors[i][: pad_start + 1]  # keep the eos token at the end

            # print("Response Tensors:", response_tensors)

            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            # print("Decoded Responses:", batch["response"])

            decoded_outputs = [q + r for q, r in zip(batch["query"], batch["response"])]
            # print("Decoded Outputs:", decoded_outputs)

            reward_vectors = []
            for value in feasible_values:
                rewards = get_reward(decoded_outputs, value, model=model_rewards.get(value), tokenizer=tokenizer_rewards.get(value), prompts=batch['query'])
                reward_vectors.append(rewards)
            # print("Reward Vectors:", reward_vectors)

            reward_matrix = torch.tensor(reward_vectors, dtype=torch.float32)
            feasible_lam = lam[feasible_values_indices]
            reward_final = torch.matmul(feasible_lam, reward_matrix)
            reward_final_list = [reward_final.unsqueeze(1)[i] for i in range(reward_final.shape[0])]
            # print("Final Reward Matrix:", reward_final_list)

            print('GPU Occupation', get_nvidia_smi_info())
    
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_final_list)
            ppo_trainer.log_stats(stats, batch, rewards)
            print("Training Stats:", stats)

        # Save the model
        save_path = convert_ppo_modelname_to_huggingface_valid(save_path)
        os.makedirs(save_path, exist_ok=True)
        ppo_trainer.save_pretrained(save_path)
        print("Training complete!")

if __name__ == "__main__":
    fire.Fire(main)




