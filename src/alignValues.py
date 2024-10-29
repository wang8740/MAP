import torch
import torch.optim as optim
import json
import fire
from utils import ALL_SUPPORTED_VALUES
import numpy as np
import csv
import os


class AlignValues:
    """A class for obtaining the most appropriate lambda values, namely linear coefficients for combining multiple rewards.

    This class handles the calculation of lambda values using a given set of (prompt, continuation) pairs generated from the reference model, and specified palette (c) if any.

    Attributes:
        c (torch.Tensor): Target palette.
        value_list (list): List of values (str) to be aligned.
        file_path (str): Path to the JSON file containing (prompt, continuation) pairs.
        rewards (torch.Tensor): Tensor of rewards for each value and sample, shape (k, n) where k is the number of values and n is the sample size in file_path.
 
    Example 1 (single human value):
        >>> c_list = -0.5
        >>> value_list = "humor"
        >>> file_path = "results/Llama27b-chat-Anthropic-harmless.json"
        >>> aligner = AlignValues(value_list, file_path, c_list)
        >>> lambda_vals, success = aligner.optimize_lambda()
        >>> print(f"Optimized lambda values: {lambda_vals}")
        >>> print(f"Optimization success: {success}")

    Example 2 (multiple human values):
        >>> c_list = [-1.016, -2.508, -1.214, -0.139, 0.848, 0.521, -1.375]
        >>> value_list = "all"
        >>> file_path = "results/Llama27b-chat-Anthropic-harmless.json"
        >>> aligner = AlignValues(value_list, file_path, c_list)
        >>> lambda_vals, success = aligner.optimize_lambda()
        >>> print(f"Optimized lambda values: {lambda_vals}")
        >>> print(f"Optimization success: {success}")

    Command-line usage:
        >>> python alignValues.py --c_list=-0.5 --value_list="humor" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda
        >>> python alignValues.py --c_list=-1.016,-2.508,-1.214,-0.139,0.848,0.521,-1.375 --value_list="all" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda
    """

    def __init__(self, value_list, file_path, c_list=None):

        """Initialize the AlignValues instance.

        Args:
            value_list (str or tuple): Values to be aligned. Can be a comma-separated string or a tuple of strings.
            file_path (str): Path to the JSON file containing (prompt, continuation) pairs.
            c_list (float or tuple, optional): Constraint value(s). Defaults to None.
        """

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
            self.rewards = torch.stack(self.rewards, dim=0) # shape [k, n] (k constraints and n samples)    

    def optimize_lambda(self, lambda_init=None, optimize_indices=None, verbose=True):

        """Optimize lambda values for the given palatte and rewards.

        This method uses gradient descent to find optimal lambda values that
        maximize the dual objective function.

        Args:
            lambda_init (list, optional): Initial lambda values. Defaults to None.
            optimize_indices (list, optional): Indices of lambda values to optimize. Defaults to None.
            verbose (bool, optional): Whether to print detailed information during optimization. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - list: Optimized lambda values.
                - bool: True if optimization was successful, False otherwise.
        """

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
       
            loss = -self._dual_objective(lambda_vals)
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
        if verbose: self._save_results_to_text(optimized_lambda, success)

        return optimized_lambda, success


    def _dual_objective(self, lambda_vals):

        # print(f"lambda_vals[:, None] * self.rewards: {lambda_vals[:, None] * self.rewards}")
        exp_terms = torch.exp(torch.sum(lambda_vals[:, None] * self.rewards, dim=0))
        mean_exp = torch.mean(exp_terms)
        
        return -torch.log(mean_exp) + torch.dot(lambda_vals, self.c)


    def sequential_optimize_lambda(self, lambda_init=None):

        """Sequentially optimize lambda for each human value.

        This method aligns each value sequentially, storing the obtained lambda values.
        It starts with lambda_init = None if not provided. Future support may replace
        optimize_indices = [idx] with block-wise updates.

        Args:
            lambda_init (list, optional): Initial lambda values. Defaults to None.

        Returns:
            list: Optimized lambda values after sequential optimization.

        Note:
            This function can be considered as a full-lambda optimization with
            freezing of values not currently being aligned.

        Example:
            >>> aligner = AlignValues("all", "results/opt1.3b-Anthropic-harmless.json", [2.513, -0.967, 0.937, 0.876, 0.434, -3.337])
            >>> optimized_lambda = aligner.sequential_optimize_lambda()
            >>> print(f"Sequentially optimized lambda: {optimized_lambda}")

        Command-line usage:
        >>> python alignValues.py --c_list=2.513,-0.967,0.937,0.876,0.434,-3.337 --value_list="all" --file_path="results/opt1.3b-Anthropic-harmless.json" sequential_optimize_lambda
        """

        for idx in range(len(self.c)):
            print(f"\n===Optimizing value {self.value_list[idx]}")
            optimize_indices = [idx]
            lambda_init, success = self.optimize_lambda(lambda_init=lambda_init, optimize_indices=optimize_indices, verbose=False)
            if not success:
                print(f"Optimization failed for value {self.value_list[idx]}")
                break

        return lambda_init


    def sequential_optimize_lambda_multiround(self, round: int = 5):
        """Run sequential_optimize_lambda for multiple rounds.

        This method performs multiple rounds of sequential lambda optimization,
        using the result of each round as the initial value for the next.

        Args:
            round (int, optional): Number of optimization rounds to perform. Defaults to 5.

        Returns:
            list: Final optimized lambda values after all rounds.

        Example:
            >>> aligner = AlignValues("all", "results/opt1.3b-Anthropic-harmless.json", [2.513, -0.967, 0.937, 0.876, 0.434, -3.337])
            >>> final_lambda = aligner.sequential_optimize_lambda_multiround(round=5)
            >>> print(f"Final optimized lambda: {final_lambda}")

        Command-line usage:
            >>> python alignValues.py --c_list=2.513,-0.967,0.937,0.876,0.434,-3.337 --value_list="all" --file_path="results/opt1.3b-Anthropic-harmless.json" sequential_optimize_lambda_multiround
        """

        # Initialize lambda with equal weights
        lambda_init=None
        for idx in range(round):
            print(f"\n\n===Running Epoch {idx}")
            lambda_init = self.sequential_optimize_lambda(lambda_init=lambda_init)

        print(f"\n\n===Final optimized lambda: {lambda_init}")
        return lambda_init

 
    def find_pareto_by_interpolation(self, c_low, c_high):
        """Automatically find the feasible palette c on the line between c_low and c_high that is closest to the Pareto frontier.

        This method uses linear interpolation to search for a feasible solution between two given constraint vectors.

        Args:
            c_low (list or float): Lower bound constraint vector or single value.
            c_high (list or float): Upper bound constraint vector or single value.

        Returns:
            float or None: The interpolation factor (rho) of the feasible solution if found, None otherwise.

        Example:
            >>> aligner = AlignValues("all", "results/basemodel-dataset.json", [2.513, -0.967, 0.937, 0.876, 0.434, -3.337])
            >>> rho = aligner.find_pareto_by_interpolation([2.513, -0.967, 0.937, 0.876, 0.434, -3.337],
            ...                                            [2.534, -0.613, 1.268, 0.876, 0.434, -3.337])
            >>> print(f"Feasible solution found at rho = {rho}")

        Command-line usage:
            >>> python alignValues.py --c_low=2.513,-0.967,0.937,0.876,0.434,-3.337 --c_high=2.534,-0.613,1.268,0.876,0.434,-3.337 --value_list="all" --file_path="results/basemodel-dataset.json" find_pareto_by_interpolation
        """

        if not isinstance(c_low, (list, tuple)):
            c_low, c_high = [c_low], [c_high]
        c_low = torch.tensor(c_low, dtype=torch.float32)
        c_high = torch.tensor(c_high, dtype=torch.float32)
        for rho in torch.linspace(0, 1, steps=20):
            self.c = c_high - rho * (c_high - c_low)
            optimized_lambda, success, = self.optimize_lambda(verbose=False)
            if success:
                print(f"Feasible solution found for rho = {rho:.3f}")
                self._save_results_to_text(optimized_lambda, success)
                return rho.item()  # Return the value of rho when a feasible solution is found

        print(f"Feasible solution not found -- neither of the input two c lists is feasible")
        return None


    def find_pareto_by_oneValue(self, value_to_enhance: str):
        """Automatically find the feasible palette c that greedily increases one particular human value closest to the Pareto frontier.

        This method uses binary search to find the maximum feasible value for a specific constraint while keeping others constant.

        Args:
            value_to_enhance (str): The name of the value to be enhanced.

        Returns:
            float: The maximum feasible value found for the enhanced constraint.

        Raises:
            ValueError: If the specified value is not in the list of supported values.

        Example:
            >>> aligner = AlignValues("all", "results/basemodel-dataset.json", [2.513, -0.967, 0.937, 0.876, 0.434, -3.337])
            >>> max_value = aligner.find_pareto_by_oneValue("gpt2-helpful")
            >>> print(f"Maximum feasible value for 'gpt2-helpful': {max_value}")

        Command-line usage:
            >>> python alignValues.py --c_list=2.513,-0.967,0.937,0.876,0.434,-3.337 --value_list="all" --value_to_enhance="gpt2-helpful" --file_path="results/basemodel-dataset.json" find_pareto_by_oneValue
        """

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
        self._save_results_to_text(optimized_lambda, success)

        return low


    def _save_results_to_text(self, optimized_lambda, success, save_prefix='results/alignValues'):
        """Save the optimization results to a text file.

        This method appends the results of lambda optimization to a text file, including
        the file path, constraint levels, values, and optimized lambda values.

        Args:
            optimized_lambda (list): List of optimized lambda values.
            success (bool): True if optimization was successful, False otherwise.
            save_prefix (str, optional): Prefix for the save file path. Defaults to 'results/alignValues'.

        Example:
            >>> aligner = AlignValues("all", "results/model-data.json", [0.5, 1.0])
            >>> optimized_lambda, success = aligner.optimize_lambda()
            >>> aligner._save_results_to_text(optimized_lambda, success)
            Results have been appended to results/alignValues.txt
        """

        c_str = ','.join(f'{v:.3f}' for v in self.c.tolist())
        optimized_lambda_str = ','.join(f'{v:.3f}' for v in optimized_lambda)
        data = f"filepath: {self.file_path}, c Levels: {c_str}, values: {self.value_list}, optimized lambda: {optimized_lambda_str if success else 'NaN'}\n"

        # Open the file in append mode and write the data
        file_path = f"{save_prefix}.txt"
        with open(file_path, 'a') as file:
            file.write(data)

        print(f"Results have been appended to {file_path}")


    def _save_results_to_csv(self, optimized_lambda, dirichlet_lambda, save_prefix='results/alignValues'):
        """Save the optimization results to a CSV file.

        This method appends the results of lambda optimization to a CSV file, including
        the file path, constraint levels, values, optimized lambda values, and Dirichlet
        reference lambda values.

        Args:
            optimized_lambda (list): List of optimized lambda values.
            dirichlet_lambda (list): List of Dirichlet reference lambda values.
            save_prefix (str, optional): Prefix for the save file path. Defaults to 'results/alignValues'.

        Example:
            >>> aligner = AlignValues("all", "results/model-data.json", [0.5, 1.0])
            >>> optimized_lambda, _ = aligner.optimize_lambda()
            >>> dirichlet_lambda = [0.3, 0.7]  # Example Dirichlet reference values
            >>> aligner._save_results_to_csv(optimized_lambda, dirichlet_lambda)
            Results have been appended to results/alignValues.csv
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


    def gen_rand_MAP_lambda(self, num_lambda: int, scaling_MAX: float, save_prefix: str = 'rand_MAP_lambda'):
        """Generate random MAP lambda values with constraints.

        This method generates random lambda values by drawing
        each c_i randomly between the current c_i and the maximum reward corresponding
        to value i. It modifies the c values, recalculates lambda, and returns a list
        of lambda values constrained by scaling_MAX.

        Args:
            num_lambda (int): Number of valid lambda values to generate.
            scaling_MAX (float): Maximum allowed L1 norm for the generated lambda values.
            save_prefix (str, optional): Prefix for the save file path. Defaults to 'rand_MAP_lambda'.

        Returns:
            tuple: A tuple containing:
                - list: Generated lambda values that satisfy the constraints.
                - float: Success rate of lambda generation attempts.

        Example:
            >>> aligner = AlignValues("all", "results/model-data.json", [0.5, 1.0, 1.5])
            >>> lambdas, success_rate = aligner.gen_rand_MAP_lambda(10, 5.0)
            >>> print(f"Generated {len(lambdas)} lambda values with a success rate of {success_rate:.2%}")

        Note:
            This method temporarily modifies the instance's c values during execution
            but restores them to their original values before returning.
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
                    
                    self._save_results_to_csv(optimized_lambda, dirichlet_lambda, save_prefix)
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
