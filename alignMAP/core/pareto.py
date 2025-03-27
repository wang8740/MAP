"""Pareto frontier optimization methods for AlignMAP.

This module contains methods for finding points on the Pareto frontier for
multi-value alignment problems. These methods allow finding optimal trade-offs
between different human values when perfect alignment is not possible.
"""

import torch
import numpy as np
import logging
from typing import List, Tuple, Optional, Union, Any, Dict

from alignmap.utils.device import get_device

logger = logging.getLogger(__name__)

class ParetoOptimizer:
    """Class for finding points on the Pareto frontier for value alignment.
    
    This class provides methods to find optimal trade-offs between multiple human values
    when perfect alignment for all values simultaneously is not possible.
    
    Attributes:
        aligner: An instance of AlignValues to use for optimization
    """
    
    def __init__(self, aligner):
        """Initialize the ParetoOptimizer with an AlignValues instance.
        
        Args:
            aligner: An AlignValues instance to use for optimization
        """
        self.aligner = aligner
    
    def find_pareto_by_interpolation(
        self, 
        c_low: Union[List[float], torch.Tensor], 
        c_high: Union[List[float], torch.Tensor]
    ) -> Tuple[float, Optional[List[float]], Optional[List[float]]]:
        """Perform bisection search to find the feasible palette c on the line between c_low and c_high.
        
        Uses a bisection search approach to find the point on the line between c_low and
        c_high that is closest to the Pareto frontier.
        
        Args:
            c_low (Union[List[float], torch.Tensor]): Lower bound constraint vector
            c_high (Union[List[float], torch.Tensor]): Upper bound constraint vector
            
        Returns:
            Tuple[float, Optional[List[float]], Optional[List[float]]]: Tuple containing:
                - success_rate: Ratio of successful optimizations
                - best_c: Best constraint vector found, or None if not successful
                - best_lambda: Best lambda values found, or None if not successful
                
        Example:
            >>> from alignmap.core.alignment import AlignValues
            >>> aligner = AlignValues(value_list="all", rewards=rewards, c_list=[0.5, 0.8, 0.3])
            >>> pareto = ParetoOptimizer(aligner)
            >>> success_rate, best_c, best_lambda = pareto.find_pareto_by_interpolation(
            ...     c_low=[0.1, 0.2, 0.1], c_high=[1.0, 1.5, 0.8]
            ... )
        """
        if not isinstance(c_low, (list, tuple, torch.Tensor)):
            c_low, c_high = [c_low], [c_high]
        
        device = self.aligner.device
        c_low = torch.tensor(c_low, dtype=torch.float32, device=device)
        c_high = torch.tensor(c_high, dtype=torch.float32, device=device)

        best_rho = None
        best_c = None
        best_lambda = None
        success_count = 0
        iteration_count = 0

        while torch.norm(c_high - c_low) > 1e-3:  # Continue until the range is sufficiently small
            iteration_count += 1
            rho = 0.5  # Midpoint
            c_mid = c_low + rho * (c_high - c_low)
            self.aligner.c = c_mid.clone()  # Ensure `c_mid` is cloned to prevent accidental overwriting

            optimized_lambda, success = self.aligner.optimize_lambda(verbose=False)
            if success:
                success_count += 1
                # Update best solution
                best_rho = rho
                best_c = c_mid.clone()  # Clone to ensure immutability
                best_lambda = optimized_lambda
                # Move c_low closer to c_mid
                c_low = c_mid.clone()  # Clone to prevent modifying `c_mid` accidentally
            else:
                # Move c_high closer to c_mid
                c_high = c_mid.clone()  # Clone to prevent modifying `c_mid` accidentally

        if best_rho is not None:
            logger.info(f"Feasible solution found with rho = {best_rho:.3f}")
            success_rate = success_count / iteration_count
            return success_rate, best_c.tolist(), best_lambda

        logger.warning("Feasible solution not found.")
        return 0, None, None
    
    def find_pareto_by_grid_search(
        self, 
        c_low: Union[List[float], torch.Tensor], 
        c_high: Union[List[float], torch.Tensor],
        grid_points: int = 10
    ) -> Tuple[float, Optional[List[float]], Optional[List[float]]]:
        """Perform grid search to find a feasible palette c from c_low to c_high.
        
        Searches through a grid of points on the line from c_low to c_high to find 
        points that are feasible for optimization.
        
        Args:
            c_low (Union[List[float], torch.Tensor]): Lower bound constraint vector
            c_high (Union[List[float], torch.Tensor]): Upper bound constraint vector
            grid_points (int): Number of grid points to check between c_low and c_high
            
        Returns:
            Tuple[float, Optional[List[float]], Optional[List[float]]]: Tuple containing:
                - success_rate: Ratio of successful optimizations
                - best_c: Best constraint vector found, or None if not successful
                - best_lambda: Best lambda values found, or None if not successful
                
        Example:
            >>> from alignmap.core.alignment import AlignValues
            >>> aligner = AlignValues(value_list="all", rewards=rewards, c_list=[0.5, 0.8, 0.3])
            >>> pareto = ParetoOptimizer(aligner)
            >>> success_rate, best_c, best_lambda = pareto.find_pareto_by_grid_search(
            ...     c_low=[0.1, 0.2, 0.1], c_high=[1.0, 1.5, 0.8], grid_points=20
            ... )
        """
        if not isinstance(c_low, (list, tuple, torch.Tensor)):
            c_low, c_high = [c_low], [c_high]
        
        device = self.aligner.device
        c_low = torch.tensor(c_low, dtype=torch.float32, device=device)
        c_high = torch.tensor(c_high, dtype=torch.float32, device=device)

        # Generate grid points between c_low and c_high
        rho_values = torch.linspace(0.0, 1.0, grid_points, device=device)
        grid = [c_low + rho * (c_high - c_low) for rho in rho_values]
        
        best_c = None
        best_lambda = None
        success_count = 0
        iteration_count = 0

        for c in grid:
            iteration_count += 1
            self.aligner.c = c.clone()  # Set current palette

            # Check feasibility
            optimized_lambda, success = self.aligner.optimize_lambda(verbose=False)
            if success:
                success_count += 1
                best_c = c.clone()  # Update best feasible solution
                best_lambda = optimized_lambda
            else:
                logger.info(f"Infeasible solution encountered at grid point {iteration_count}. Stopping search.")
                break  # Stop further grid search if infeasibility is encountered

        if best_c is not None:
            logger.info(f"Feasible solution found after {success_count} feasible grid points.")
            success_rate = success_count / iteration_count
            return success_rate, best_c.tolist(), best_lambda

        logger.warning("No feasible solution found.")
        return 0, None, None
    
    def find_pareto_by_one_value(
        self, 
        value_to_enhance: str,
        precision: float = 0.05
    ) -> float:
        """Find the maximum feasible value for a specific constraint.
        
        Automatically find the feasible palette c that greedily increases one particular 
        human value closest to the Pareto frontier.
        
        Args:
            value_to_enhance (str): The name of the value to be enhanced
            precision (float): Precision for binary search
            
        Returns:
            float: The maximum feasible value found for the enhanced constraint
            
        Raises:
            ValueError: If the specified value is not in the aligner's value list
            
        Example:
            >>> from alignmap.core.alignment import AlignValues
            >>> aligner = AlignValues(value_list=["humor", "helpfulness", "harmlessness"], 
            ...                       rewards=rewards, c_list=[0.5, 0.8, 0.3])
            >>> pareto = ParetoOptimizer(aligner)
            >>> max_value = pareto.find_pareto_by_one_value("helpfulness")
            >>> print(f"Maximum feasible value for helpfulness: {max_value}")
        """
        if value_to_enhance not in self.aligner.value_list:
            raise ValueError(f"{value_to_enhance} is not in the list of supported values.")
        
        dimension = self.aligner.value_list.index(value_to_enhance)
        original_value = self.aligner.c[dimension].item()  # Store the original value to restore later
        
        # Determine the range for adjustment
        low = original_value  # Start from the current value
        high = torch.max(self.aligner.rewards[dimension]).item()  # Max value from the rewards for this dimension

        # Binary search for the maximum feasible value
        while high - low > precision:  # Continue until the range is sufficiently small
            mid = (low + high) / 2
            self.aligner.c[dimension] = mid
            _, success = self.aligner.optimize_lambda(verbose=False)
            if success:
                low = mid  # If feasible, increase lower bound
            else:
                high = mid  # If not feasible, decrease upper bound
        
        # Set to the maximum feasible value and optimize
        self.aligner.c[dimension] = low
        optimized_lambda, success = self.aligner.optimize_lambda(verbose=False)
        
        if success:
            logger.info(f"Enhanced {value_to_enhance} from {original_value:.3f} to {low:.3f}")
        else:
            logger.warning(f"Could not enhance {value_to_enhance} beyond {original_value:.3f}")
            low = original_value

        return low


# Convenience functions for more direct usage

def find_pareto_by_interpolation(
    aligner,
    c_low: Union[List[float], torch.Tensor], 
    c_high: Union[List[float], torch.Tensor]
) -> Tuple[float, Optional[List[float]], Optional[List[float]]]:
    """Convenience function to find Pareto frontier point by interpolation.
    
    See ParetoOptimizer.find_pareto_by_interpolation for details.
    """
    optimizer = ParetoOptimizer(aligner)
    return optimizer.find_pareto_by_interpolation(c_low, c_high)

def find_pareto_by_grid_search(
    aligner,
    c_low: Union[List[float], torch.Tensor], 
    c_high: Union[List[float], torch.Tensor],
    grid_points: int = 10
) -> Tuple[float, Optional[List[float]], Optional[List[float]]]:
    """Convenience function to find Pareto frontier point by grid search.
    
    See ParetoOptimizer.find_pareto_by_grid_search for details.
    """
    optimizer = ParetoOptimizer(aligner)
    return optimizer.find_pareto_by_grid_search(c_low, c_high, grid_points)

def find_pareto_by_one_value(
    aligner,
    value_to_enhance: str,
    precision: float = 0.05
) -> float:
    """Convenience function to find maximum feasible value for one constraint.
    
    See ParetoOptimizer.find_pareto_by_one_value for details.
    """
    optimizer = ParetoOptimizer(aligner)
    return optimizer.find_pareto_by_one_value(value_to_enhance, precision) 