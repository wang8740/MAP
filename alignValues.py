import torch
import torch.optim as optim
import json
import fire
from utils import ALL_SUPPORTED_VALUES
import numpy as np
import csv
import os


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
            self.rewards = torch.stack(self.rewards, dim=0) # dimension is k x n

        # print(f"stacked self.rewards is {self.rewards}")

    def optimize_lambda(self, lambda_init=None, optimize_indices=None, verbose=True):

        print("\nRunning AlignValues.optimize_lambda")

        # Initial tau=log(lambda) values
        if lambda_init is None:
            lambda_vals = torch.zeros_like(self.c, requires_grad=False)
            lambda_vals[optimize_indices] = 1.0  # Initialize the selected indices and keeping the rest at zero
        else:
            lambda_vals = torch.tensor(lambda_init, requires_grad=False)
            lambda_vals[optimize_indices] = 1.0  # Initialize the selected indices and keeping the rest as is (in case some values had been aligned)

        # Check if optimize_indices is provided, else optimize all
        if optimize_indices is None:
            optimize_indices = list(range(len(self.c)))

        # Set up tau_optimizable based on selected indices
        tau_optimizable = torch.tensor([torch.log(lambda_vals[i]).item() for i in optimize_indices], requires_grad=True)
        
        optimizer = optim.Adam([tau_optimizable], lr=0.1)

        # Optimization loop
        success = True
        for step in range(150):
            optimizer.zero_grad()
        
            # Update lambda_vals based on tau_optimizable
            lambda_vals[optimize_indices] = torch.exp(tau_optimizable)
       
            loss = -self.dual_objective(lambda_vals)
            if verbose: print(f"Loss = {loss}")

            if torch.any(torch.isnan(lambda_vals)) or torch.any(torch.isinf(lambda_vals)):
                if verbose: print("Lambda values diverged to NaN or Inf.")
                success = False
                break

            # loss.backward()
            loss.backward(retain_graph=True)

            optimizer.step()

            if step % 50 == 0 and verbose:
                print(f"Step {step}: Tau = {tau_optimizable.tolist()}, Lambda = {lambda_vals.tolist()}, Loss = {loss.item()}")

        if success:
            optimized_lambda = lambda_vals.tolist()
            print(f"\nOptimized lambda: {','.join(f'{v:.3f}' for v in optimized_lambda)}")
        else:
            optimized_lambda = ['NaN']
            print(f"\nOptimized lambda: NaN")

        # Save results to Excel
        if verbose: self.save_results_to_text(optimized_lambda, success)

        return optimized_lambda, success


    def dual_objective(self, lambda_vals):

        # print(f"lambda_vals[:, None] * self.rewards: {lambda_vals[:, None] * self.rewards}")
        exp_terms = torch.exp(torch.sum(lambda_vals[:, None] * self.rewards, dim=0))
        mean_exp = torch.mean(exp_terms)
        
        return -torch.log(mean_exp) + torch.dot(lambda_vals, self.c)

    # NOTE: we do not need this function as we can essentially treat it as the full-lambda optimization but freezing those who are not currently being aligned
    # def dual_objective_weighted(self, tau_vals, lambda_prev, c_current):

    #     lambda_vals = torch.exp(tau_vals)

    #     # Assuming lambda_prev is already in the exponential form and self.rewards is [k, n]
    #     weights = torch.softmax(torch.sum(lambda_prev[:, None] * self.rewards, dim=0), dim=0)  # Apply softmax over samples
        
    #     exp_terms = torch.exp(torch.sum(lambda_vals[:, None] * self.rewards, dim=0))  # Sum over constraints for each sample
        
    #     weighted_exp = weights * exp_terms
        
    #     return -torch.log(torch.sum(weighted_exp)) + torch.dot(lambda_vals, c_current)


    def sequential_optimize_lambda(self, lambda_init=None):

        for idx in range(len(self.c)):
            print(f"\n===Optimizing value {self.value_list[idx]}")
            optimize_indices = [idx]
            lambda_init, success = self.optimize_lambda(lambda_init=lambda_init, optimize_indices=optimize_indices, verbose=False)
            if not success:
                print(f"Optimization failed for value {self.value_list[idx]}")
                break

        return lambda_init
    # Example:
    # python alignValues.py --c_list=2.513,-0.967,0.937,0.876,0.434,-3.337 --value_list="all" --file_path="results/opt1.3b-Anthropic-harmless.json" optimize_lambda
    # python alignValues.py --c_list=2.513,-0.967,0.937,0.876,0.434,-3.337 --value_list="all" --file_path="results/opt1.3b-Anthropic-harmless.json" sequential_optimize_lambda
    # ideal result: 12.766,1.526,1.689,0.012,0.019,0.023

    def sequential_optimize_lambda_multiround(self):

        # Initialize lambda with equal weights
        round = 5
        lambda_init=None
        for idx in range(round):
            print(f"\n\n===Running Epoch {idx}")
            lambda_init = self.sequential_optimize_lambda(lambda_init=lambda_init)

        print(f"\n\n===Final optimized lambda: {lambda_init}")
        return lambda_init
    # Example:
    # python alignValues.py --c_list=2.513,-0.967,0.937,0.876,0.434,-3.337 --value_list="all" --file_path="results/opt1.3b-Anthropic-harmless.json" sequential_optimize_lambda_multiround
    # optimized result after 5 rounds: [12.564545631408691, 1.4926655292510986, 1.6635115146636963, 0.01948617585003376, 0.02005678042769432, 0.0251987986266613]

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
        # Enhance a particular value so that it reaches maximal possible 
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


    def save_results_to_text(self, optimized_lambda, success, save_prefix='results/alignValues'):
        """
         Save the results to text file. This is used to generate the results file and save it to disk
         
         :param optimized_lambda: list of optimized lambda values
         :param success: True if success False if failure ( NaN in case of failure
        """
        c_str = ','.join(f'{v:.3f}' for v in self.c.tolist())
        optimized_lambda_str = ','.join(f'{v:.3f}' for v in optimized_lambda)
        data = f"filepath: {self.file_path}, c Levels: {c_str}, values: {self.value_list}, optimized lambda: {optimized_lambda_str if success else 'NaN'}\n"

        # Open the file in append mode and write the data
        file_path = f"{save_prefix}.txt"
        with open(file_path, 'a') as file:
            file.write(data)

        print(f"Results have been appended to {file_path}")


    def save_results_to_csv(self, optimized_lambda, dirichlet_lambda, save_prefix='results/alignValues'):
        """
        Save the results to a CSV file. This function appends new data each time it's called.
        
        :param optimized_lambda: list of optimized lambda values
        :param dirichlet_lambda: list of Dirichlet reference lambda values
        :param save_prefix: prefix for the save file path
        """
        file_path = f"{save_prefix}.csv"
        
        # Prepare the data
        c_str = ','.join(f'{v:.3f}' for v in self.c.tolist())
        optimized_lambda_str = ','.join(f'{v:.3f}' for v in optimized_lambda)
        dirichlet_lambda_str = ','.join(f'{v:.3f}' for v in dirichlet_lambda)
        
        # Prepare the row data
        row_data = [
            self.file_path,  # filepath
            c_str,  # c Levels
            ','.join(self.value_list),  # values
            optimized_lambda_str,  # optimized lambda
            dirichlet_lambda_str,  # Dirichlet lambda reference
        ]
        
        # Prepare the header
        header = [
            'filepath',
            'c_Levels',
            'values',
            'optimized_lambda',
            'Dirichlet_lambda_ref'
        ]
        
        # Check if the file exists and is empty
        file_exists = os.path.isfile(file_path)
        file_empty = os.stat(file_path).st_size == 0 if file_exists else True

        # Open in append mode
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write the header only if the file is new or empty
            if file_empty:
                csvwriter.writerow(header)
            
            # Write the data
            csvwriter.writerow(row_data)
        
        print(f"Results have been appended to {file_path}")


    def gen_rand_MAP_lambda(self, num_lambda, scaling_MAX, save_prefix='rand_MAP_lambda'):
        """
        Generate random MAP lambda values by drawing each c_i randomly between the current c_i
        and the maximum reward corresponding to value i. This function modifies the c values,
        recalculates lambda, and returns a list of lambda values constrained by scaling_MAX.
        
        :param num_lambda: Number of valid lambda values to generate
        :param scaling_MAX: Maximum allowed L1 norm for the generated lambda values
        :return: Tuple containing list of generated lambda values and success rate
        """
        generated_lambdas = []
        total_attempts = 0
        successful_attempts = 0

        # Store original c values to restore later
        original_c = self.c.clone()

        # Continue until we have the specified number of valid lambda values
        while len(generated_lambdas) < num_lambda:
            total_attempts += 1
            
            # Draw new c values randomly between current c and maximum rewards
            for i in range(len(self.c)):
                max_reward = torch.max(self.rewards[i]).item()  # Get the maximum reward for the ith value
                self.c[i] = torch.tensor(np.random.uniform(original_c[i].item(), max_reward), dtype=torch.float32)
            
            # Optimize lambda with the new c values
            optimized_lambda, success = self.optimize_lambda(verbose=False)
            
            if success:
                # Check if the L1 norm of optimized_lambda is within the scaling_MAX constraint
                if sum(x for x in optimized_lambda) <= scaling_MAX:
                    generated_lambdas.append(optimized_lambda)
                    successful_attempts += 1

                    # Generate the Dirichlet reference lambda
                    random_alpha = np.random.dirichlet(np.ones(len(self.c)), 1)[0]
                    random_lam = np.random.uniform(0, scaling_MAX) * random_alpha
                    dirichlet_lambda = random_lam.tolist()
                    
                    self.save_results_to_csv(optimized_lambda, dirichlet_lambda, save_prefix)
                    print(f"Valid lambda found. Random c: {self.c.tolist()}, Optimized lambda: {optimized_lambda}, Dirichlet_lambda_ref: {dirichlet_lambda}")
                else:
                    print(f"Invalid lambda. L1 norm exceeds scaling_MAX. Random c: {self.c.tolist()}, Optimized lambda: {optimized_lambda}")
            else:
                print(f"Invalid lambda. Random c: {self.c.tolist()}, Optimization failed")

        # Restore the original c values
        self.c = original_c

        # Calculate success rate
        success_rate = successful_attempts / total_attempts

        print(f"\nGenerated {num_lambda} valid lambda values.")
        print(f"Success rate: {success_rate:.2%} ({successful_attempts} successes out of {total_attempts} attempts)")

        return generated_lambdas, success_rate
    

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