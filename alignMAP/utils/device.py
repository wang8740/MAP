"""Utility functions for device handling (CPU/GPU/etc.)."""

import os
import torch
import logging
from typing import Optional, Union, List

logger = logging.getLogger(__name__)

def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for PyTorch operations.
    
    Args:
        device (Optional[str]): Device specification ('cpu', 'cuda', 'cuda:0', etc.)
            If None, will use CUDA if available, otherwise CPU.
            
    Returns:
        torch.device: PyTorch device object
        
    Raises:
        ValueError: If the specified device is invalid or not available
    """
    # Use specified device if provided
    if device is not None:
        # Check if valid CUDA device
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                logger.warning(f"CUDA requested but not available. Falling back to CPU.")
                return torch.device('cpu')
            
            # Handle specific GPU selection
            if ':' in device:
                device_id = int(device.split(':')[1])
                if device_id >= torch.cuda.device_count():
                    available_gpus = torch.cuda.device_count()
                    logger.warning(
                        f"GPU {device_id} requested but only {available_gpus} GPUs available. "
                        f"Using device 'cuda:0' instead."
                    )
                    return torch.device('cuda:0')
            
            return torch.device(device)
        
        # Handle other device types
        return torch.device(device)
    
    # Auto-detect: use CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available, using CPU")
    
    return device

def get_device_info() -> dict:
    """Get information about the available devices.
    
    Returns:
        dict: Dictionary containing device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
        'current_device': 'cpu'
    }
    
    if info['cuda_available']:
        info['device_count'] = torch.cuda.device_count()
        for i in range(info['device_count']):
            info['devices'].append({
                'name': torch.cuda.get_device_name(i),
                'capability': torch.cuda.get_device_capability(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            })
        
        info['current_device'] = f"cuda:{torch.cuda.current_device()}"
    
    return info

def is_gpu_available() -> bool:
    """Check if a GPU is available for training.
    
    Returns:
        bool: True if a GPU is available, False otherwise
    """
    return torch.cuda.is_available()

def set_device_environment(device: Optional[str] = None) -> torch.device:
    """Set environment variables for optimal performance on the given device.
    
    Args:
        device (Optional[str]): Device specification
            
    Returns:
        torch.device: The configured PyTorch device
    """
    # Get the device object
    device_obj = get_device(device)
    
    # Optimize environment based on device type
    if device_obj.type == 'cuda':
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark for potentially faster training
        torch.backends.cudnn.benchmark = True
        
        # Set environment variables for optimal GPU performance
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings with HF tokenizers
    else:
        # Set environment variables for optimal CPU performance
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    
    return device_obj

def get_optimal_device_settings(model_size: Optional[str] = None) -> dict:
    """Get optimal device settings based on model size and available hardware.
    
    Args:
        model_size (Optional[str]): Size category of the model ('small', 'medium', 'large')
            
    Returns:
        dict: Dictionary with optimal settings for the device
    """
    settings = {
        'device': get_device(),
        'precision': 'float32',
        'gradient_accumulation_steps': 1,
        'mixed_precision': False,
    }
    
    # Determine if GPU is available and get memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        settings['available_memory'] = gpu_mem
        
        # Enable mixed precision for GPUs with enough memory
        if gpu_mem >= 8:
            settings['mixed_precision'] = True
            settings['precision'] = 'float16'
        
        # Adjust gradient accumulation based on model size and available memory
        if model_size == 'large' and gpu_mem < 16:
            settings['gradient_accumulation_steps'] = 4
        elif model_size == 'medium' and gpu_mem < 8:
            settings['gradient_accumulation_steps'] = 2
    
    return settings

def get_device_count() -> int:
    """Get the number of available devices.
    
    Returns:
        int: Number of available GPU devices, or 1 for CPU/MPS
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def set_device_settings(
    device: str, 
    mixed_precision: bool = False,
    memory_efficient: bool = False
) -> None:
    """Configure device-specific settings for optimal performance.
    
    Args:
        device (str): Device to configure
        mixed_precision (bool): Whether to use mixed precision training
        memory_efficient (bool): Whether to use memory-efficient settings
    """
    if device.startswith("cuda"):
        # Set CUDA-specific optimizations
        if mixed_precision:
            logger.info("Enabling mixed precision for CUDA")
            # Set up amp for PyTorch native mixed precision
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        if memory_efficient:
            logger.info("Enabling memory-efficient settings for CUDA")
            # Free memory cache
            torch.cuda.empty_cache()
            
            # Set environment variables for memory efficiency if not already set
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    elif device == "mps":
        logger.info("Configuring for MPS device")
        # MPS-specific settings could go here
    
    elif device == "cpu":
        logger.info("Configuring for CPU device")
        # CPU-specific settings
        torch.set_num_threads(max(1, os.cpu_count() or 1))


def get_optimal_batch_size(
    device: str,
    model_size_in_billions: float,
    sequence_length: int = 512,
    target_memory_usage: float = 0.7
) -> int:
    """Estimate an optimal batch size based on device and model.
    
    Args:
        device (str): The device to use
        model_size_in_billions (float): Model size in billions of parameters
        sequence_length (int): Maximum sequence length to process
        target_memory_usage (float): Target memory usage as fraction of available
    
    Returns:
        int: Estimated optimal batch size
    """
    # Constants for estimation
    bytes_per_parameter = 4  # 32-bit float
    bytes_per_optimization_state = 8  # Adam uses two states per parameter
    
    # Memory needed for model and optimization state (in GB)
    model_memory = model_size_in_billions * bytes_per_parameter
    optim_memory = model_size_in_billions * bytes_per_optimization_state
    
    # Memory per sample (approximate)
    memory_per_sample = 2 * sequence_length * model_size_in_billions * 1e-9
    
    if device.startswith("cuda"):
        # Get available GPU memory
        device_idx = 0 if device == "cuda" else int(device.split(":")[-1])
        torch.cuda.set_device(device_idx)
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)  # in GB
        
        # Calculate available memory
        reserved_memory = model_memory + optim_memory
        available_memory = (total_memory - reserved_memory) * target_memory_usage
        
        # Calculate batch size
        batch_size = max(1, int(available_memory / memory_per_sample))
        
    elif device == "mps":
        # For Apple Silicon, use a more conservative estimate
        # These values are approximations
        batch_size = max(1, int(8 / (memory_per_sample * 2)))
        
    else:  # CPU
        # For CPU, batch size depends more on compute than memory
        # Using a simple heuristic based on model size
        batch_size = max(1, int(4 / model_size_in_billions))
    
    return batch_size 