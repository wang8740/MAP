import torch
import torch.optim as optim
import json
import fire
from utils import ALL_SUPPORTED_VALUES

class AlignValues:
    def __init__(self, value_list, file_path, c_list=None):

        # file_path is the json file that contains the pretrained model-generated responses for numerical calculation of lambda given c values

        # c should be 1D with length k (number of constraints)
        # rewards should be 2D with shape [k, n] (k constraints and n samples)      
          
        # print(f"input c_list is {c_list} (type {type(c_list)}), value_list is {value_list} (type {type(value_list)})")
        # Note: for 1D input, the types are float, str), and for 2D input, the types are (tuple, tuple)

        # Load constraints. Make it a list if it's a single scalar
        if c_list:
            if not isinstance(c_list, (list, tuple)):
                c_list = [c_list]
            self.c = torch.tensor(c_list, dtype=torch.float32)
        else:
            self.c = None
        
        # Load values. If more than one value make sure it is a list
        if isinstance(value_list, str):
            if value_list == "all":
                self.value_list = ALL_SUPPORTED_VALUES
            else:
                self.value_list = value_list.split(',')  # Split the string into a list
        # print(f"type of value_list is {type(value_list)}, value_list is {value_list}")
   
        self.file_path = file_path
        if self.c is not None:  print(f"Input c is {self.c} for aligning {value_list}, using json file {self.file_path}")

        # Load data
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        
        if len(self.value_list) == 1:
            value = self.value_list[0]
            print(f"value_list: {self.value_list}, value: {value}")
            value_rewards = [entry[value] for entry in data]
            tensor_rewards = torch.tensor(value_rewards, dtype=torch.float32)
            self.rewards = tensor_rewards.unsqueeze(0)  # Make 2D if only one dimension
            print(f"Average {value} reward: {torch.mean(tensor_rewards).item():.4f}")

        else:
            self.rewards = []
            for value in self.value_list:
                value_rewards = [entry[value] for entry in data]
                tensor_rewards = torch.tensor(value_rewards, dtype=torch.float32)
                self.rewards.append(tensor_rewards)
                print(f"Average {value} reward: {torch.mean(tensor_rewards).item():.4f}")
            self.rewards = torch.stack(self.rewards, dim=0)

        # print(f"stacked self.rewards is {self.rewards}")

    def optimize_lambda(self, verbose=True):

        print("\nRunning AlignValues.optimize_lambda")

        # Initial tau values
        tau_init = torch.zeros_like(self.c, requires_grad=True)
        optimizer = optim.Adam([tau_init], lr=0.1)

        # Optimization loop
        success = True
        for step in range(150):
            optimizer.zero_grad()
            loss = -self.dual_objective(tau_init)
            if verbose: print(f"Loss = {loss}")

            lambda_vals = torch.exp(tau_init)
            if torch.any(torch.isnan(lambda_vals)) or torch.any(torch.isinf(lambda_vals)):
                if verbose: print("Lambda values diverged to NaN or Inf.")
                success = False
                break

            loss.backward()
            
            if step == 0 and verbose:
                print(f"Initial gradients: {tau_init.grad}") 

            optimizer.step()

            if step % 50 == 0 and verbose:
                print(f"Step {step}: Tau = {tau_init.tolist()}, Lambda = {lambda_vals.tolist()}, Loss = {loss.item()}")

        if success:
            optimized_lambda = torch.exp(tau_init).tolist()
            print(f"Optimized lambda: {','.join(f'{v:.3f}' for v in optimized_lambda)}")
        else:
            optimized_lambda = ['NaN']

        # Save results to Excel
        if verbose: self.save_results_to_text(optimized_lambda, success)

        return optimized_lambda, success


    def dual_objective(self, tau_vals):

        lambda_vals = torch.exp(tau_vals)
        # print(f"lambda_vals[:, None] * self.rewards: {lambda_vals[:, None] * self.rewards}")
        exp_terms = torch.exp(torch.sum(lambda_vals[:, None] * self.rewards, dim=0))
        mean_exp = torch.mean(exp_terms)
        
        return -torch.log(mean_exp) + torch.dot(lambda_vals, self.c)


    def find_pareto_by_interpolation(self, c_low, c_high):

        if not isinstance(c_low, (list, tuple)):
            c_low, c_high = [c_low], [c_high]
        c_low = torch.tensor(c_low, dtype=torch.float32)
        c_high = torch.tensor(c_high, dtype=torch.float32)
        for rho in torch.linspace(0, 1, steps=20):
            self.c = c_high - rho * (c_high - c_low)
            optimized_lambda, success, = self.optimize_lambda(verbose=False)
            if success:
                print(f"Feasible solution found for rho = {rho:.3f}")
                self.save_results_to_text(optimized_lambda, success)
                return rho.item()  # Return the value of rho when a feasible solution is found

        print(f"Feasible solution not found -- neither of the input two c lists is feasible")
        return None


    def find_pareto_by_oneValue(self, value_to_enhance):
        # Find the index of the value to enhance in the value list
        if value_to_enhance not in self.value_list:
            raise ValueError(f"{value_to_enhance} is not in the list of supported values.")
        
        dimension = self.value_list.index(value_to_enhance)
        original_value = self.c[dimension].item()  # Store the original value to restore later
        
        # Determine the range for adjustment
        low = original_value  # Start from the current value
        high = torch.max(self.rewards[dimension]).item()  # Max value from the rewards for this dimension

        # Binary search for the maximum feasible value
        while high - low > 0.05:  # Continue until the range is sufficiently small
            mid = (low + high) / 2
            self.c[dimension] = mid
            _, success = self.optimize_lambda(verbose=False)
            if success:
                low = mid  # If feasible, increase lower bound
            else:
                high = mid  # If not feasible, decrease upper bound
        
        self.c[dimension] = low
        optimized_lambda, success, = self.optimize_lambda(verbose=False)
        print(f"Enhanced {value_to_enhance} from {original_value:.3f} to {low:.3f}")
        self.save_results_to_text(optimized_lambda, success)

        return low


    def save_results_to_text(self, optimized_lambda, success):
        """
         Save the results to text file. This is used to generate the results file and save it to disk
         
         :param optimized_lambda: list of optimized lambda values
         :param success: True if success False if failure ( NaN in case of failure
        """
        file_path = 'results/alignValues.txt' 
        c_str = ','.join(f'{v:.3f}' for v in self.c.tolist())
        optimized_lambda_str = ','.join(f'{v:.3f}' for v in optimized_lambda)
        data = f"filepath: {self.file_path}, c Levels: {c_str}, values: {self.value_list}, optimized lambda: {optimized_lambda_str if success else 'NaN'}\n"

        # Open the file in append mode and write the data
        with open(file_path, 'a') as file:
            file.write(data)

        print(f"Results have been appended to {file_path}")



if __name__ == '__main__':
    fire.Fire(AlignValues)

# Command line usage examples
# For a single constraint or value
# python alignValues.py --c_list=-0.5 --value_list="humor" --file_path="results/Llama27b-chat-Anthropic-harmless.json"

# For multiple constraints or values
# python alignValues.py --c_list=-0.5,-0.5 --value_list="humor,humor" --file_path="results/Llama27b-chat-Anthropic-harmless.json"

# To optimize
# python alignValues.py --c_list=-0.5 --value_list="humor" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda

# !python alignValues.py --c_list=-1.016,-2.508,-1.214,-0.139,0.848,0.521,-1.375 --value_list="all" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda