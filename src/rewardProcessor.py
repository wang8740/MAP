import json
from utils import get_reward, get_model_and_tokenizer, ALL_SUPPORTED_VALUES
import fire
import torch
import numpy as np
# from typing import List, AnyStr
import os
import pandas as pd
from shutil import copyfile


class RewardProcessor:
    """Processes reward values for a model output evaluation.

    This class is used to evaluate and align different values (such as diversity, coherence, etc.)
    for a set of generated model outputs.

    Attributes:
        values_to_evaluate (Optional[List[str]]): List of values to evaluate.
        values_to_align (Optional[List[str]]): List of values to align.
        file_path (str): Path to the JSON file containing generated outputs.
        batch_size (int): Batch size for processing rewards.
    """

    def __init__(self, values_to_evaluate=None, values_to_align=None, file_path=None, batch_size=32):
        """Initializes the RewardProcessor with parameters for evaluation and alignment.

        Args:
            values_to_evaluate (Optional[Union[str, List[str]]]): Values to evaluate, either as a
                comma-separated string or list. Defaults to None.
            values_to_align (Optional[Union[str, List[str]]]): Values to align, either as a
                comma-separated string or list. Defaults to None.
            file_path (Optional[str]): Path to the JSON file to process. Defaults to None.
            batch_size (int): Size of batches to process at a time. Defaults to 32.
        """
        print(f"\nRunning RewardProcessor.__init__ for {file_path}\n")

        self.values_to_align_str = None
        self.values_to_evaluate_str = None

        if values_to_evaluate: 
            if isinstance(values_to_evaluate, str):
                if values_to_evaluate == "all":
                    self.values_to_evaluate = ALL_SUPPORTED_VALUES
                    self.values_to_evaluate_str = "all"
                else:
                    # self.values_to_evaluate = [values_to_evaluate]
                    self.values_to_evaluate = values_to_evaluate.split(',')  # Split the string into a list
                    self.values_to_evaluate_str  = values_to_evaluate
            else:
                self.values_to_evaluate = list(values_to_evaluate)
        else:
            self.values_to_evaluate = None
        
        if values_to_align: 
            if isinstance(values_to_align, str):
                if values_to_align == "all":
                    self.values_to_align = ALL_SUPPORTED_VALUES
                    self.values_to_align_str = "all"
                else:
                    # self.values_to_align = [values_to_align]
                    self.values_to_align = values_to_align.split(',')
                    self.values_to_align_str  = values_to_align
            else:
                self.values_to_align = list(values_to_align)
        else:
            self.values_to_align = None

        self.file_path = file_path
        self.batch_size = batch_size
        
    
    def add_reward(self, value, basemodel_for_perplexity=None):
        """Adds a specific reward to the dataset in a non-invasive manner.

        Args:
            value (str): The reward type to add (e.g., "diversity", "perplexity").
            basemodel_for_perplexity (Optional[str]): Base model required for "perplexity" value.
                Defaults to None.

        Raises:
            ValueError: If `value` is "perplexity" and `basemodel_for_perplexity` is not provided.

        # Example usage to add reward, often via submitting parallel pbs files to accelerate computation:
        >>> reward_processor = RewardProcessor(file_path="results/Llama27b-chat-Anthropic-harmless.json")
        >>> reward_processor.add_reward(value="humor")

        # Command-line usage:
        >>> python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --value="humor" add_reward
        """
        if value == "perplexity" and not basemodel_for_perplexity:
            raise ValueError("basemodel_for_perplexity, which should be the data-gen model, must be provided for perplexity task")

        temp_folder = os.path.join(os.path.dirname(self.file_path), "temp")
        # remove '.json' before appending temp information
        base_name = os.path.basename(self.file_path).rsplit('.', 1)[0]
        temp_file_path = os.path.join(temp_folder, f"{base_name}_temp_{value}.json")

        # Ensure the temp folder exists
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        
        # Copy the original file to the temporary file
        copyfile(self.file_path, temp_file_path)
        
        print(f"Running add_reward for value: {value} and temporarily save to {temp_file_path}")
    
        # Load existing data from the JSON file
        with open(temp_file_path, 'r') as file:
            data = json.load(file)

        # Load the model and tokenizer
        if value not in ["diversity", "perplexity"]:
            model, tokenizer = get_model_and_tokenizer(value)            
        elif value == "perplexity":
            model_generate, tokenizer_generate = get_model_and_tokenizer(basemodel_for_perplexity)
        else:
            pass

        # Process each batch and update the data
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_sentences = [entry["generated"] for entry in batch]
            batch_prompts = [entry["prompt"] for entry in batch]

            if value == "diversity":
                rewards = get_reward(batch_sentences, value)
            elif value == "positive":
                # if sentiment control task, use log prob for interpretability
                rewards = get_reward(batch_sentences, value, model=model, tokenizer=tokenizer, use_score=False) 
            elif value == "perplexity":
                rewards = get_reward(batch_sentences, value, model=model_generate, tokenizer=tokenizer_generate)
            else:
                rewards = get_reward(batch_sentences, value, model=model, tokenizer=tokenizer, prompts=batch_prompts)
            
            # Update each entry with the computed reward
            for entry, reward in zip(batch, rewards):
                entry[value] = reward

            # Save updated data back to the file
            with open(temp_file_path, 'w') as file:
                json.dump(data, file, indent=4)
        return

    def quantile_transform_single_c(self, c_list):
        """Transforms a list of c values into quantiles.

        Args:
            c_list (List[float]): List of c values to transform.

        Returns:
            List[float]: List of quantile values.
        """
        print(f"\nRunning RewardProcessor.quantile_transform_single_c\n")

        with open(self.file_path, 'r') as file:
            data = json.load(file)
        quant = []
        for i, value in enumerate(ALL_SUPPORTED_VALUES):
            list_rewards = np.array([entry[value] for entry in data])
            sorted_rewards = np.sort(list_rewards)
            quant.append(np.searchsorted(sorted_rewards, c_list[i]) / len(sorted_rewards))
        res = f"quant: {','.join(f'{q:.3f}' for q in quant)}"
        print(res)
        return quant

    def assess_original_value(self, evaluation_mode = False):
        """Assesses the original level of each value in the dataset.

        Args:
            evaluation_mode (bool): If True, calculates quantiles; otherwise, calculates only average. Defaults to False.
    
        # Example usage to get realized values or c-levels under the original model:
        >>> reward_processor = RewardProcessor(file_path="results/Llama27b-chat-Anthropic-harmless.json", values_to_evaluate="all")
        >>> reward_processor.assess_original_value()

        # As a natural followup, one could define custom `c_align` values, e.g., set c to be 20% improvement:
        >>> import numpy as np
        >>> c_noalign = [-1.239, -2.731, -1.437, -0.362, 0.848, 0.521, -1.375]
        >>> c_align = [x + np.log(1.25) for x in c_noalign[:4]]
        >>> print(f"c_align: {','.join(f'{v:.3f}' for v in c_align)}") # [-1.016,-2.508,-1.214,-0.139,0.848,0.521,-1.375]

        # Command-line usage:
        >>> python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_evaluate="all" assess_original_value

        """
        print(f"\nRunning RewardProcessor.assess_original_value\n")

        # assess the original level of value, defined as the expected reward under the original data distribution (namely the one that generated the data stored in the file)
        with open(self.file_path, 'r') as file:
            data = json.load(file)

        # Prepare DataFrame to store all statistics
        statistics = {'Statistic': ['avg', 'avg_std', '50%', '60%', '70%', '80%', '90%', '99%']}
        maintained_values = {'diversity', 'coherence', 'perplexity'}

        for value in self.values_to_evaluate:
            list_rewards = np.array([entry[value] for entry in data if value in entry])

            # Calculate average and quantiles
            if list_rewards.size > 0:
                avg_reward = np.mean(list_rewards)
                std_error = np.std(list_rewards) / np.sqrt(len(list_rewards))
                if evaluation_mode:
                    quantiles = np.percentile(list_rewards, [50, 60, 70, 80, 90, 99])
                    statistics[value] = [avg_reward, std_error] + [q for q in quantiles]

                else:
                    # for the sake of determining c-values
                    if value in maintained_values:
                        # Use average for all quantile rows for maintained values
                        statistics[value] = [avg_reward, std_error] + [avg_reward] * 6
                    else:
                        quantiles = np.percentile(list_rewards, [50, 60, 70, 80, 90, 99])
                        max_of_avg_and_quantiles = [max(avg_reward, q) for q in quantiles]
                        statistics[value] = [avg_reward, std_error] + max_of_avg_and_quantiles
            else:
                # Handle case with no data
                statistics[value] = [0] * 8

        # Convert dictionary to DataFrame
        print(f"\nstatistics: \n{statistics}")
        df = pd.DataFrame(statistics).round(3)
        # csv_path = "results/original_rewards.csv"
        csv_path = self.file_path.replace(".json", "_realC-levels.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved in {csv_path}")

        return

    def _assess_postalignment_singlevalue(self, singlevalue_to_evaluate, lam, debug=True):
        """Assesses a single value's alignment level after applying alignment weights, 
        as weight-approximated by data stored in the file originally generated from pre-alignment distribution.

        Args:
            singlevalue_to_evaluate (str): The value to evaluate alignment for.
            lam (Union[float, List[float]]): Alignment weights.
            debug (bool): If True, prints debugging information. Defaults to True.

        Returns:
            float: Estimated alignment level for the single value.
        """

        if not isinstance(lam, (list, tuple)):
            lam_list = [lam]
        elif not isinstance(lam, list):
            lam_list = list(lam)
        else:
            lam_list = lam
        lam_tensor = torch.tensor(lam_list, dtype=torch.float32)
        
        if debug: print(f"\nUnder lam={lam_list} aligned for values={self.values_to_align}, we assess the level of value {singlevalue_to_evaluate}")

        # Load data
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        
        if len(self.values_to_align) == 1:
            v = self.values_to_align[0]
            value_rewards = [entry[v] for entry in data]
            tensor_rewards = torch.tensor(value_rewards, dtype=torch.float32)
            rewards = tensor_rewards.unsqueeze(0)  # Make 2D if only one dimension

        else:
            rewards = []
            for v in self.values_to_align:
                value_rewards = [entry[v] for entry in data]
                tensor_rewards = torch.tensor(value_rewards, dtype=torch.float32)
                rewards.append(tensor_rewards)
            rewards = torch.stack(rewards, dim=0)

        # Compute the softmax of lambda times rewards
        weights = torch.softmax(torch.sum(lam_tensor[:, None] * rewards, dim=0), dim=0)
        if debug: print(f"\nNumber of weighted samples: {weights.size()}")
        
        rewards_of_value_to_eval = torch.tensor([entry[singlevalue_to_evaluate] for entry in data], dtype=torch.float32)
        estimated_c = torch.sum(weights * rewards_of_value_to_eval).item()

        if debug: print(f"\nUsing weighted average of rewards to estimate c for {singlevalue_to_evaluate} to be: {estimated_c}")

        return estimated_c
    
    def assess_postalignment_multivalue(self, lam=None, k=100, scaling=1.0, scaling_MAX=1):
        """
        Applies `assess_postalignment_singlevalue` across multiple values and alignments.
        If lam is not given, assess the c level by a random vector of lam drawn from the probability simplex multiplied by the scaling factor
        If scaling < 0, then randomly select scaling factor from a range.
        If lam is given, overwrite the k and scaling to simply use the given lam.

        Args:
            lam (Optional[Union[float, List[float]]]): Fixed alignment weights. Defaults to None.
            k (int): Number of random samples for Monte Carlo. Defaults to 100.
            scaling (float): Scaling factor for random lambda. Defaults to 1.0.
            scaling_MAX (int): Maximum scaling for random lambda. Defaults to 1.


        # Example 1 
        # Usage for post-alignment assessment of multiple values:
        >>> reward_processor = RewardProcessor(file_path="results/Llama27b-chat-Anthropic-harmless.json", values_to_align="humor,harmless", lam=[0.41, 0.37], values_to_evaluate="all")
        >>> reward_processor.assess_postalignment_multivalue()

        # Command-line usage:
        >>> python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="humor,harmless" --lam=0.41,0.37 --values_to_evaluate="all" assess_postalignment_multivalue
        >>> python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="humor,harmless" --lam=0.41,0.37 --values_to_evaluate="humor,harmless" assess_postalignment_multivalue

        # Example 2 
        # Pareto frontier study with random lambda (often used in conjunction with plot_pareto.py to visualize the Pareto frontier)
        >>> reward_processor = RewardProcessor(file_path="results/Llama27b-chat-Anthropic-harmless.json", values_to_align="humor,harmless", values_to_evaluate="all", scaling=-1)
        >>> reward_processor.assess_postalignment_multivalue()

        # Command-line usage:
        >>> python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="humor,harmless" --values_to_evaluate="all" --scaling=-1 assess_postalignment_multivalue

        """
        print(f"\nRunning RewardProcessor.assess_postalignment_multivalue\n")

        if lam:
            k = 1
            if not isinstance(lam, (list, tuple)):
                lam_list = [lam]
            elif not isinstance(lam, list):
                lam_list = list(lam)
            else:
                lam_list = lam
            # print(f"\nlam_list: {lam_list}, type: {type(lam_list)}, type of lam_list[0]: {type(lam_list[0])}")
            print(f"\nUsing lam={lam_list} to align values={self.values_to_align}, we assess the c-level of values {self.values_to_evaluate}")
            lam_str = ','.join(f'{float(l):.3f}' for l in lam_list)
            csv_path = self.file_path.replace(".json", f"_lam={lam_str}_val={self.values_to_align_str}_numericC-levels.csv")
        else:
            print(f"\nUsing random lam to align values={self.values_to_align}, we assess the c-level of values {self.values_to_evaluate}")
            csv_path = self.file_path.replace(".json", f"_pareto{scaling}_randLam_val={self.values_to_align_str}_numericC-levels.csv")

        # use Monte Carlo to estimate the c
        m = len(self.values_to_align)  # Number of dimensions in the probability simplex
        results = []
        for _ in range(k):
            # Generate a random point in the probability simplex
            random_alpha = np.random.dirichlet(np.ones(m), 1)[0]
            # print(f"random_alpha: {random_alpha}")
            if lam:
                random_lam = lam_list
            elif scaling < 0:
                random_lam = np.random.uniform(0, scaling_MAX) * random_alpha
                random_lam = random_lam.tolist()
                # random_lam[1] *= 15  # for diversity
            else:
                random_lam = scaling * random_alpha
                random_lam = random_lam.tolist()

            estimated_c = []
            for v in self.values_to_evaluate:
                estimated_c.append(self._assess_postalignment_singlevalue(v, random_lam, debug=False))
            results.append(estimated_c + random_lam)        
        
        # Create a new DataFrame and save
        df = pd.DataFrame(results, columns=self.values_to_evaluate + [v + "_lam" for v in self.values_to_align])
        df = df.round(3)

        max_values = df.max()
        print("\nCompute the maximum of each column:")
        for col, max_value in max_values.items():
            print(f"{col}: {max_value:.3f}")

        df.to_csv(csv_path, index=False)
        print(f"Results saved/updated in {csv_path}")

        return


if __name__ == '__main__':
    fire.Fire(RewardProcessor)