import json
from utils import get_reward, get_model_and_tokenizer, ALL_SUPPORTED_VALUES
import fire
import torch
import numpy as np
# from typing import List, AnyStr
import os
import pandas as pd
from shutil import copyfile

def genGaussian(n, rho, colnames, sd=0.5):
    # Input the number of samples k and the desired correlation rho

    # Generate k samples from a standard normal distribution
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    cov_matrix = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov_matrix)
    z = np.stack((x, y), axis=0)
    correlated_data = np.dot(L, z) * sd

    return pd.DataFrame(correlated_data.T, columns=colnames)

class RewardProcessor:
    def __init__(self, values_to_evaluate=None, values_to_align=None, file_path=None, batch_size=32):

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
        '''
            The process is non-invasive, meaning that if the original json files has existing rewards for values not equal to value, it will not be overwritten or emptied
        '''
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

        print(f"\nRunning RewardProcessor.quantile_transform_single_c\n")

        with open(self.file_path, 'r') as file:
            data = json.load(file)
        quant = []
        for i, value in enumerate(ALL_SUPPORTED_VALUES):
            list_rewards = np.array([entry[value] for entry in data])
            sorted_rewards = np.sort(list_rewards)
            quant.append(np.searchsorted(sorted_rewards, c_list[i]) / len(sorted_rewards))
        print(f"quant: {','.join(f'{q:.3f}' for q in quant)}")
        return 

    def assess_original_value(self, evaluation_mode = False):

        print(f"\nRunning RewardProcessor.assess_original_value\n")

        # assess the original level of value, defined as the expected reward under the original data distribution (namely the one that generated the data stored in the file)
        with open(self.file_path, 'r') as file:
            data = json.load(file)

        # average_rewards = []
        # # Extract all values and calculate the average
        # for value in self.values_to_evaluate:
        #     list_rewards = [entry[value] for entry in data]
        #     average_reward = sum(list_rewards) / len(list_rewards) if list_rewards else 0
        #     print(f"Average {value} reward: {average_reward:.3f}")
        #     average_rewards.append(average_reward)
        
        # df = pd.DataFrame([average_rewards], columns=self.values_to_evaluate).round(3)
        # csv_path = f"results/original_rewards.csv"
        # df.to_csv(csv_path, index=False)
        # print(f"Results saved in {csv_path}")

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
        # lam should be in List format
        # assess the level (c) of singlevalue_to_evaluate after alignment, as weight-approximated by data stored in the file originally generated from pre-alignment distribution
        # namely the weighted average of the reward where weight is propto exponential lambda times the reward of aligned values
        # aligned_values: list of values that are aligned, lambda: corresponding weights
        # both value and elements in aligned_values should appear in the dataset

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

        print(f"\nRunning RewardProcessor.assess_postalignment_multivalue\n")
        # apply assess_postalignment_singlevalue to various lam drawn from the prob simplex
        # if lam is not given, assess the c level by a random vector of lam drawn from the probability simplex multiplied by the scaling factor
        # if scaling < 0, then randomly select scaling factor from a range
        # if lam is given, overwrite the k and scaling to simply use the given lam

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
                random_lam[1] *= 15  # for diversity
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

        # use dummy df for ablation study
        # df = genGaussian(n=k, rho=0.0, colnames=self.values_to_evaluate, sd=0.5)

        max_values = df.max()
        print("\nCompute the maximum of each column:")
        for col, max_value in max_values.items():
            print(f"{col}: {max_value:.3f}")

        df.to_csv(csv_path, index=False)
        print(f"Results saved/updated in {csv_path}")

        return


if __name__ == '__main__':
    fire.Fire(RewardProcessor)


'''
    RUN to calculate rewards, often with parallel pbs: 
    !python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --value="humor" add_reward
'''


'''
    RUN to get the original rewards, namely the c level, before alignment
    !python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_evaluate="all" assess_original_value
'''


'''
    USE alignValues.py to get the lambda from user-defined c
    !python alignValues.py --c_list=-0.005,-0.485,-0.078,0.848,0.521,-1.375 --value_list="all" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda

'''


# set c to be 20% improvement
# import numpy as np
# c_noalign = [-1.239,-2.731,-1.437,-0.362,0.848,0.521,-1.375]
# c_align = [x + np.log(1.25) for x in c_noalign[:4]]
# print(f"c_align: {','.join(f'{v:.3f}' for v in c_align)}")
# c_align: -1.016,-2.508,-1.214,-0.139,0.848,0.521,-1.375
# lam = [0.174,0.017,1.264,9.201,0.004,0.141,0.014]

# After optimizing lambda for "humor" from c we record
# (c_humor, lambda_humor): (-2, 0.00), (-1.3 0.01), (-1.2, 0.02), (-0.1, 1.29), (0.5, NA)--goes to inf
# for the pair (-0.1, 1.29), we generated aligned datasets and calculated Average humor reward: -0.1521 and harmless reward: -2.7116, which is successful! 

# After optimizing lambda for "humor,harmless" from c we record
 # c=(-0.5,-1.5) -> lambda=(0.41, 0.37)


'''
    RUN to numerically estimate the c level for all values after alignment
    ALTERNATIVELY, we can realistically generate samples using inference-stage MC approach or training-stage PPO approach
    !python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="humor,harmless" --lam=0.41,0.37 --values_to_evaluate="all" assess_postalignment_multivalue
    !python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="all" --lam=321.402,6.117,29.249,0.003,1.108,0.718 --values_to_evaluate="all" assess_postalignment_multivalue

    Or simply for selected values: 
    !python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="humor,harmless" --lam=0.41,0.37 --values_to_evaluate="humor,harmless" assess_postalignment_multivalue
'''

# --values_to_align="humor,harmless" --lam=0.41,0.37
# -0.48788657784461975,-2.1130435466766357,-1.50656259059906,-0.2896653413772583,0.8501017689704895,0.5168927907943726,-1.3379247188568115,0.41,0.37

# --values_to_align="all" --lam=see below
# humor,harmless,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity,   humor_lam,harmless_lam,gpt2-helpful_lam,gpt2-harmless_lam,diversity_lam,coherence_lam,perplexity_lam
# -1.016,-1.914,-1.214,-0.139,0.861,0.521,-1.252,     0.174,0.017,1.264,9.201,0.004,0.141,0.014


'''
    RUN this get random lambda for Pareto study
    !python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="humor,harmless" --values_to_evaluate="all" --scaling=-1 assess_postalignment_multivalue
    !python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="all" --values_to_evaluate="all" --scaling=-1 assess_postalignment_multivalue

    USE together with plot_pareto.py to visualize the Pareto frontier
'''
