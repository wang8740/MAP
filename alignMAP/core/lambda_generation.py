"""Lambda generation methods for AlignMAP.

This module contains methods for generating lambda values for multi-value alignment,
including random generation with constraints and result saving.
"""

import torch
import numpy as np
import csv
import os
import json
import logging
from typing import List, Tuple, Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

class LambdaGenerator:
    """Class for generating lambda values for value alignment.
    
    This class provides methods to generate lambda values for multi-value alignment,
    including random generation with constraints.
    
    Attributes:
        aligner: An instance of AlignValues to use for optimization
    """
    
    def __init__(self, aligner):
        """Initialize the LambdaGenerator with an AlignValues instance.
        
        Args:
            aligner: An AlignValues instance to use for optimization
        """
        self.aligner = aligner
    
    def gen_rand_MAP_lambda(
        self, 
        num_lambda: int, 
        scaling_MAX: float,
        save_prefix: str = 'rand_MAP_lambda',
        save_path: Optional[str] = None
    ) -> Tuple[List[List[float]], float]:
        """Generate random MAP lambda values with constraints.

        This method generates random lambda values by drawing
        each c_i randomly between the current c_i and the maximum reward corresponding
        to value i. It modifies the c values, recalculates lambda, and returns a list
        of lambda values constrained by scaling_MAX.

        Args:
            num_lambda (int): Number of valid lambda values to generate
            scaling_MAX (float): Maximum allowed L1 norm for the generated lambda values
            save_prefix (str): Prefix for the save file path
            save_path (Optional[str]): Path to save results to
            
        Returns:
            Tuple[List[List[float]], float]: Tuple containing:
                - list of generated lambda values that satisfy the constraints
                - success rate of lambda generation attempts
                
        Example:
            >>> from alignmap.core.alignment import AlignValues
            >>> aligner = AlignValues(value_list=["humor", "helpfulness", "harmlessness"],
            ...                       rewards=rewards, c_list=[0.5, 0.8, 0.3])
            >>> generator = LambdaGenerator(aligner)
            >>> lambdas, success_rate = generator.gen_rand_MAP_lambda(10, 5.0)
            >>> print(f"Generated {len(lambdas)} lambda values with a success rate of {success_rate:.2%}")
        """
        generated_lambdas = []
        total_attempts = 0
        successful_attempts = 0

        # Store original c values to restore later
        original_c = self.aligner.c.clone()

        # Continue until we have the specified number of valid lambda values
        while len(generated_lambdas) < num_lambda:
            total_attempts += 1
            
            # Draw new c values randomly between current c and maximum rewards
            for i in range(len(self.aligner.c)):
                max_reward = torch.max(self.aligner.rewards[i]).item()  # Get the maximum reward for the ith value
                self.aligner.c[i] = torch.tensor(
                    np.random.uniform(original_c[i].item(), max_reward), 
                    dtype=torch.float32,
                    device=self.aligner.device
                )
            
            # Optimize lambda with the new c values
            optimized_lambda, success = self.aligner.optimize_lambda(verbose=False)
            
            if success:
                # Check if the L1 norm of optimized_lambda is within the scaling_MAX constraint
                if sum(x for x in optimized_lambda) <= scaling_MAX:
                    generated_lambdas.append(optimized_lambda)
                    successful_attempts += 1

                    # Generate the Dirichlet reference lambda
                    random_alpha = np.random.dirichlet(np.ones(len(self.aligner.c)), 1)[0]
                    random_lam = np.random.uniform(0, scaling_MAX) * random_alpha
                    dirichlet_lambda = random_lam.tolist()
                    
                    # Save results
                    if save_path:
                        self._save_results_to_csv(
                            optimized_lambda, 
                            dirichlet_lambda, 
                            save_prefix=save_prefix,
                            save_path=save_path
                        )
                    
                    logger.info(
                        f"Valid lambda found. Random c: {self.aligner.c.tolist()}, "
                        f"Optimized lambda: {optimized_lambda}, "
                        f"Dirichlet_lambda_ref: {dirichlet_lambda}"
                    )
                else:
                    logger.debug(
                        f"Invalid lambda. L1 norm exceeds scaling_MAX. "
                        f"Random c: {self.aligner.c.tolist()}, "
                        f"Optimized lambda: {optimized_lambda}"
                    )
            else:
                logger.debug(f"Invalid lambda. Random c: {self.aligner.c.tolist()}, Optimization failed")

        # Restore the original c values
        self.aligner.c = original_c

        # Calculate success rate
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0

        logger.info(f"Generated {num_lambda} valid lambda values.")
        logger.info(f"Success rate: {success_rate:.2%} ({successful_attempts} successes out of {total_attempts} attempts)")

        return generated_lambdas, success_rate
    
    def _save_results_to_csv(
        self, 
        optimized_lambda: List[float], 
        dirichlet_lambda: List[float],
        save_prefix: str = 'results/alignValues',
        save_path: Optional[str] = None
    ) -> None:
        """Save optimization results to a CSV file.
        
        Args:
            optimized_lambda (List[float]): Optimized lambda values
            dirichlet_lambda (List[float]): Reference Dirichlet lambda values
            save_prefix (str): Prefix for the save file path
            save_path (Optional[str]): Path to save results to
        """
        # Default save path
        if save_path is None:
            save_path = f"{save_prefix}_{len(self.aligner.c)}-values_results.csv"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Prepare the data
        c_str = ','.join(f'{v:.3f}' for v in self.aligner.c.tolist())
        optimized_lambda_str = ','.join(f'{v:.3f}' for v in optimized_lambda)
        dirichlet_lambda_str = ','.join(f'{v:.3f}' for v in dirichlet_lambda)
        
        # Prepare the row data
        row_data = [
            getattr(self.aligner, 'file_path', 'N/A'),  # filepath
            c_str,  # c Levels
            ','.join(self.aligner.value_list),  # values
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
        file_exists = os.path.isfile(save_path)
        file_empty = os.stat(save_path).st_size == 0 if file_exists else True

        # Open in append mode
        with open(save_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write the header only if the file is new or empty
            if file_empty:
                csvwriter.writerow(header)
            
            # Write the data
            csvwriter.writerow(row_data)
        
        logger.info(f"Results have been appended to {save_path}")


# Convenience functions for direct usage

def gen_rand_MAP_lambda(
    aligner,
    num_lambda: int, 
    scaling_MAX: float,
    save_prefix: str = 'rand_MAP_lambda',
    save_path: Optional[str] = None
) -> Tuple[List[List[float]], float]:
    """Convenience function to generate random MAP lambda values.
    
    See LambdaGenerator.gen_rand_MAP_lambda for details.
    """
    generator = LambdaGenerator(aligner)
    return generator.gen_rand_MAP_lambda(num_lambda, scaling_MAX, save_prefix, save_path)

def save_lambda_results_to_csv(
    aligner,
    optimized_lambda: List[float], 
    dirichlet_lambda: List[float],
    save_prefix: str = 'results/alignValues',
    save_path: Optional[str] = None
) -> None:
    """Convenience function to save lambda optimization results to CSV.
    
    See LambdaGenerator._save_results_to_csv for details.
    """
    generator = LambdaGenerator(aligner)
    generator._save_results_to_csv(optimized_lambda, dirichlet_lambda, save_prefix, save_path) 