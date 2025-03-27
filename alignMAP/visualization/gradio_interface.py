"""Gradio interface for the AlignMAP tool."""

import logging
import os
import json
import torch
import numpy as np
import gradio as gr

from typing import List, Dict, Any, Tuple, Optional, Union

from alignmap.core.alignment import AlignValues
from alignmap.utils.device import get_device

logger = logging.getLogger(__name__)

def retrieve_rewards_min_max_avg(
    rewards: torch.Tensor, 
    values_to_align: List[str]
) -> Dict[str, Dict[str, float]]:
    """Calculate min, max, and average for each value in the dataset.
    
    Args:
        rewards (torch.Tensor): A tensor of shape (k, n) where k is the number of values 
                               and n is the number of samples
        values_to_align (list): List of value names corresponding to the rows in rewards
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary with value statistics
    """
    rewards_np = rewards.cpu().numpy()
    stats = {}
    for idx, value in enumerate(values_to_align):
        stats[value] = {
            'min': float(np.min(rewards_np[idx])),
            'max': float(np.max(rewards_np[idx])),
            'avg': float(np.mean(rewards_np[idx])),
        }
    return stats


def estimate_realized_levels(
    lam: Union[List[float], np.ndarray], 
    rewards: torch.Tensor
) -> torch.Tensor:
    """Estimate the realized levels for all values using the given lambda.
    
    Args:
        lam (List[float] or np.ndarray): A vector of weights matching the number of rows in rewards
        rewards (torch.Tensor): A tensor of shape (k, n) where k is the number of values 
                               and n is the number of samples
                               
    Returns:
        torch.Tensor: A tensor of realized levels (size k) as the weighted sum for each value
    """
    # Convert lambda to a tensor and ensure dimensions match
    lam_tensor = torch.tensor(lam, dtype=torch.float32, device=rewards.device)
    assert lam_tensor.size(0) == rewards.size(0), "Lambda dimension does not match the number of values."
    
    # Compute weights for each sample based on the softmax of the weighted sum of rewards
    weights = torch.softmax(torch.sum(lam_tensor[:, None] * rewards, dim=0), dim=0)
    
    # Compute realized levels for each value as a weighted sum across samples
    realized_levels = torch.matmul(rewards, weights)  # Shape (k,)
    return realized_levels


class GradioMAPInterface:
    """Gradio interface for AlignMAP."""
    
    def __init__(self, server_name: str = "0.0.0.0", share: bool = False):
        """Initialize the Gradio interface.
        
        Args:
            server_name (str): Server name or IP address to bind to
            share (bool): Whether to create a public link for the interface
        """
        self.server_name = server_name
        self.share = share
        self.interface = None
        
        # Default supported values
        self.values_to_align = [
            "humor", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"
        ]
        self.values_to_align_names = [
            "Humor", "Helpfulness", "Harmlessness", "Diversity", "Coherence", "Perplexity"
        ]
        
        # Default file paths for models with pre-computed rewards
        self.file_paths = {
            "Llama2-7B": "results/llama2_chat-Anthropic-harmless.json",
            "OPT-1.3B": "results/opt1.3b-Anthropic-harmless.json",
        }
        
        # Initialize AlignValues instances and slider settings
        self._initialize_aligners()
        
    def _initialize_aligners(self):
        """Initialize AlignValues instances for each model."""
        self.aligners = {}
        self.slider_stats = {}
        
        for model, file_path in self.file_paths.items():
            if os.path.exists(file_path):
                try:
                    aligner = AlignValues(value_list=self.values_to_align, rewards_data=file_path)
                    self.aligners[model] = aligner
                    self.slider_stats[model] = retrieve_rewards_min_max_avg(
                        aligner.rewards, self.values_to_align
                    )
                    logger.info(f"Initialized aligner for {model} from {file_path}")
                except Exception as e:
                    logger.error(f"Error initializing aligner for {model}: {e}")
            else:
                logger.warning(f"File not found: {file_path}")
    
    def optimize_lambda(
        self, 
        model_choice: str, 
        *c_values: List[float]
    ) -> Tuple[str, List[float]]:
        """Optimize lambda values for the given model and palette.
        
        Args:
            model_choice (str): Model to use
            *c_values (List[float]): Target palette values
            
        Returns:
            Tuple[str, List[float]]: Status message and realized values
        """
        logger.info(f"Optimizing lambda for model {model_choice} with palette {c_values}")
        
        if model_choice is None or model_choice not in self.aligners:
            return "Please select a valid model", [0] * len(c_values)
            
        aligner = self.aligners[model_choice]
        aligner.c = torch.tensor(c_values, dtype=torch.float32, device=aligner.device)
        
        # Try direct optimization
        lam, success = aligner.optimize_lambda(verbose=False)
        
        if success:
            realized_levels = estimate_realized_levels(lam, aligner.rewards)
            realized_levels = [float(level) for level in realized_levels]
            return f"Optimization successful! Lambda values: {lam}", realized_levels
        else:
            # If direct optimization fails, try interpolation
            c_low = [stat["avg"] for stat in self.slider_stats[model_choice].values()]
            c_high = c_values
            
            # Ensure c_high is a list
            if not isinstance(c_high, list):
                c_high = list(c_high)
                
            adjust_success, adjust_c, lam = aligner.find_pareto_by_interpolation(c_low, c_high)
            
            if adjust_success:
                realized_levels = estimate_realized_levels(lam, aligner.rewards)
                realized_levels = [float(level) for level in realized_levels]
                adjusted_c_str = ', '.join(f'{v:.3f}' for v in adjust_c)
                return f"Your specified palette is infeasible. Adjusted to feasible palette: [{adjusted_c_str}]", realized_levels
            else:
                return "Optimization failed. Try a different palette.", list(c_values)
    
    def create_sliders(self, model: str) -> List[gr.Slider]:
        """Create sliders for the given model.
        
        Args:
            model (str): Model to create sliders for
            
        Returns:
            List[gr.Slider]: List of sliders
        """
        if model not in self.slider_stats:
            # Create default sliders if stats not available
            return [
                gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label=fr"c_{i + 1}: {self.values_to_align_names[i]}",
                    interactive=True
                ) for i in range(len(self.values_to_align))
            ]
            
        return [
            gr.Slider(
                minimum=stat["min"],
                maximum=stat["max"],
                value=stat["avg"],
                step=0.1,
                label=fr"c_{i + 1}: {self.values_to_align_names[i]}",
                interactive=True
            ) for i, stat in enumerate(self.slider_stats[model].values())
        ]
    
    def update_sliders(self, model: str) -> List[Dict]:
        """Update sliders for the given model.
        
        Args:
            model (str): Model to update sliders for
            
        Returns:
            List[Dict]: List of slider update dictionaries
        """
        if model is None or model not in self.slider_stats:
            return [gr.Slider.update(value=0.5) for _ in range(len(self.values_to_align))]
            
        return [
            gr.Slider.update(
                minimum=stat["min"],
                maximum=stat["max"],
                value=stat["avg"],
                label=fr"c_{i + 1}: {self.values_to_align_names[i]}"
            ) for i, stat in enumerate(self.slider_stats[model].values())
        ]
    
    def reset_sliders(self, model_choice: str) -> List[Dict]:
        """Reset sliders to average values.
        
        Args:
            model_choice (str): Model to reset sliders for
            
        Returns:
            List[Dict]: List of slider update dictionaries
        """
        return self.update_sliders(model_choice)
    
    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface.
        
        Returns:
            gr.Blocks: Gradio interface
        """
        with gr.Blocks(title="AlignMAP Tool") as interface:
            gr.Markdown("# AlignMAP Palette Tool")
            gr.Markdown("""
            Use this tool to find optimal λ values for a target palette.
            The realized values show the actual value levels achieved by the optimization.
            """)
            
            model_choice = gr.Dropdown(
                choices=list(self.aligners.keys()), 
                value=list(self.aligners.keys())[0] if self.aligners else None, 
                label="Choose Model"
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Target Palette")
                    input_sliders = self.create_sliders(
                        list(self.aligners.keys())[0] if self.aligners else None
                    )
                    
                with gr.Column():
                    gr.Markdown("### Realized Values")
                    output_sliders = self.create_sliders(
                        list(self.aligners.keys())[0] if self.aligners else None
                    )
                    for slider in output_sliders:
                        slider.interactive = False
            
            result_box = gr.Textbox(label="Optimization Results")
            
            with gr.Row():
                optimize_btn = gr.Button("Optimize Lambda")
                reset_btn = gr.Button("Reset Values")
            
            # Set up event handlers
            inputs = [model_choice] + input_sliders
            outputs = [result_box] + output_sliders
            
            model_choice.change(
                fn=self.update_sliders,
                inputs=model_choice,
                outputs=input_sliders
            )
            
            optimize_btn.click(
                fn=self.optimize_lambda,
                inputs=inputs,
                outputs=outputs
            )
            
            reset_btn.click(
                fn=self.reset_sliders,
                inputs=[model_choice],
                outputs=input_sliders
            )
            
            self.interface = interface
            return interface
    
    def launch(self) -> str:
        """Launch the Gradio interface.
        
        Returns:
            str: Public URL if share=True, otherwise local URL
        """
        if self.interface is None:
            self.build_interface()
            
        url = self.interface.launch(
            server_name=self.server_name, 
            share=self.share
        )
        
        return url


def launch_gradio_interface(
    server_name: str = "0.0.0.0", 
    share: bool = False,
    file_paths: Optional[Dict[str, str]] = None
) -> str:
    """Launch the Gradio interface for AlignMAP.
    
    Args:
        server_name (str): Server name to bind to
        share (bool): Whether to create a public link
        file_paths (Optional[Dict[str, str]]): Custom file paths for models
        
    Returns:
        str: URL of the launched interface
    """
    interface = GradioMAPInterface(server_name=server_name, share=share)
    
    # Override default file paths if provided
    if file_paths:
        interface.file_paths = file_paths
        interface._initialize_aligners()
        
    return interface.launch() 