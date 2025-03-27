"""Registry for reward models in alignmap."""

from typing import Dict, Type, Callable, Optional, List, Any, Union
import logging
from functools import wraps

from alignmap.models.reward_models.base import BaseRewardModel, FunctionRewardModel
from alignmap.utils.device import get_device

logger = logging.getLogger(__name__)

# Global registry of reward models
_REWARD_MODEL_REGISTRY: Dict[str, Type[BaseRewardModel]] = {}

def register_reward_model(name: Optional[str] = None):
    """Decorator to register a reward model class.
    
    Args:
        name (Optional[str]): Name to register the model under.
            If None, uses the class name.
            
    Returns:
        Callable: Decorator function
    
    Example:
        @register_reward_model("my_model")
        class MyRewardModel(BaseRewardModel):
            ...
    """
    def decorator(cls):
        model_name = name or cls.__name__
        if not issubclass(cls, BaseRewardModel):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseRewardModel")
        
        if model_name in _REWARD_MODEL_REGISTRY:
            logger.warning(f"Reward model {model_name} already registered. Overwriting.")
        
        _REWARD_MODEL_REGISTRY[model_name] = cls
        logger.info(f"Registered reward model: {model_name}")
        return cls
    
    return decorator


def register_reward_function(name: str):
    """Decorator to register a function as a reward model.
    
    Args:
        name (str): Name to register the function under
        
    Returns:
        Callable: Decorator function
    
    Example:
        @register_reward_function("my_function_model")
        def my_reward_fn(texts, prompts, kwargs):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Create a FunctionRewardModel and register it
        class_name = f"{name.capitalize()}FunctionModel"
        model_cls = type(class_name, (FunctionRewardModel,), {})
        
        @register_reward_model(name)
        class _(FunctionRewardModel):
            def __init__(self, device=None):
                super().__init__(name, func, device)
        
        return wrapper
    
    return decorator


def list_available_reward_models() -> List[str]:
    """List all registered reward models.
    
    Returns:
        List[str]: List of registered model names
    """
    return list(_REWARD_MODEL_REGISTRY.keys())


def get_reward_model(
    name: str,
    device: Optional[str] = None,
    **kwargs
) -> BaseRewardModel:
    """Get a reward model by name.
    
    Args:
        name (str): Name of the reward model
        device (Optional[str]): Device to run the model on
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        BaseRewardModel: Instantiated reward model
        
    Raises:
        ValueError: If the model name is not registered
    """
    device = get_device(device)
    
    if name not in _REWARD_MODEL_REGISTRY:
        raise ValueError(
            f"Reward model '{name}' not found. "
            f"Available models: {list_available_reward_models()}"
        )
    
    model_cls = _REWARD_MODEL_REGISTRY[name]
    model = model_cls(device=device, **kwargs)
    
    return model


def create_composite_reward_model(
    models: Dict[str, Union[str, BaseRewardModel]],
    weights: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
    name: str = "composite_model"
) -> BaseRewardModel:
    """Create a composite reward model from multiple models.
    
    Args:
        models (Dict[str, Union[str, BaseRewardModel]]): Mapping of model names to
            either model names (to be loaded) or already instantiated models
        weights (Optional[Dict[str, float]]): Weights for each model
        device (Optional[str]): Device to run the models on
        name (str): Name for the composite model
        
    Returns:
        BaseRewardModel: Composite reward model
    """
    device = get_device(device)
    
    # Initialize models if needed
    instantiated_models = {}
    for model_name, model in models.items():
        if isinstance(model, str):
            instantiated_models[model_name] = get_reward_model(model, device=device)
        else:
            instantiated_models[model_name] = model
    
    # Use equal weights if not specified
    if weights is None:
        weights = {name: 1.0 / len(instantiated_models) for name in instantiated_models}
    
    # Create composite reward function
    def composite_reward_fn(texts, prompts, kwargs):
        weighted_rewards = []
        
        for model_name, model in instantiated_models.items():
            model_rewards = model.calculate_reward(texts, prompts=prompts, **kwargs)
            weighted_model_rewards = [r * weights[model_name] for r in model_rewards]
            weighted_rewards.append(weighted_model_rewards)
        
        # Sum rewards across models
        composite_rewards = [sum(rewards) for rewards in zip(*weighted_rewards)]
        return composite_rewards
    
    # Return a FunctionRewardModel with the composite function
    return FunctionRewardModel(name, composite_reward_fn, device)


# Import default models to register them
# This will be added later when we implement the specific reward models 