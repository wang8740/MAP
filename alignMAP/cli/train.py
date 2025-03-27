"""Command-line interface for training models with alignment techniques."""

import os
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any

from alignmap.training import train_ppo, train_dpo, train_dpo_with_reward_model
from alignmap.models.reward_models import list_available_reward_models
from alignmap.utils.device import get_device, is_gpu_available

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Set up logging configuration.
    
    Args:
        verbose (bool): Whether to use DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.
    
    Args:
        args (Optional[List[str]]): Command line arguments
        
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train language models with alignment techniques"
    )
    
    # Add subparsers for different training methods
    subparsers = parser.add_subparsers(dest="command", help="Training method")
    
    # PPO training parser
    ppo_parser = subparsers.add_parser("ppo", help="Train with PPO")
    ppo_parser.add_argument("--model", type=str, required=True, 
                           help="Model name or path (e.g., 'gpt2', 'facebook/opt-1.3b')")
    ppo_parser.add_argument("--reward-models", type=str, required=True,
                           help="Comma-separated list of reward model names, or 'all'")
    ppo_parser.add_argument("--lambda-values", type=str, default=None,
                           help="Comma-separated list of lambda values for reward models")
    ppo_parser.add_argument("--dataset", type=str, default="Anthropic-harmless",
                           help="Dataset name to use for training")
    ppo_parser.add_argument("--learning-rate", type=float, default=1e-6,
                           help="Learning rate for training")
    ppo_parser.add_argument("--batch-size", type=int, default=20,
                           help="Batch size for training")
    ppo_parser.add_argument("--mini-batch-size", type=int, default=2,
                           help="Mini-batch size for optimization steps")
    ppo_parser.add_argument("--epochs", type=int, default=1,
                           help="Number of training epochs")
    ppo_parser.add_argument("--save-path", type=str, default=None,
                           help="Path to save the trained model")
    ppo_parser.add_argument("--device", type=str, default=None,
                           help="Device to run training on (e.g., 'cpu', 'cuda:0')")
    ppo_parser.add_argument("--verbose", action="store_true",
                           help="Enable verbose logging")
    
    # DPO training parser
    dpo_parser = subparsers.add_parser("dpo", help="Train with DPO")
    dpo_parser.add_argument("--model", type=str, required=True,
                           help="Model name or path (e.g., 'gpt2', 'facebook/opt-1.3b')")
    dpo_parser.add_argument("--dataset", type=str, required=True,
                           help="Dataset name with preference pairs")
    dpo_parser.add_argument("--beta", type=float, default=0.1,
                           help="Temperature parameter for DPO loss")
    dpo_parser.add_argument("--learning-rate", type=float, default=5e-5,
                           help="Learning rate for training")
    dpo_parser.add_argument("--output-dir", type=str, default=None,
                           help="Directory to save the model")
    dpo_parser.add_argument("--batch-size", type=int, default=4,
                           help="Batch size for training")
    dpo_parser.add_argument("--epochs", type=int, default=1,
                           help="Number of training epochs")
    dpo_parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                           help="Number of steps to accumulate gradients")
    dpo_parser.add_argument("--device", type=str, default=None,
                           help="Device to run training on (e.g., 'cpu', 'cuda:0')")
    dpo_parser.add_argument("--verbose", action="store_true",
                           help="Enable verbose logging")
    
    # DPO with reward model parser
    dpo_rm_parser = subparsers.add_parser("dpo-rm", help="Train with DPO using a reward model")
    dpo_rm_parser.add_argument("--model", type=str, required=True,
                             help="Model name or path (e.g., 'gpt2', 'facebook/opt-1.3b')")
    dpo_rm_parser.add_argument("--reward-model", type=str, required=True,
                             help="Reward model name to use for generating preferences")
    dpo_rm_parser.add_argument("--dataset", type=str, required=True,
                             help="Dataset name to generate prompts from")
    dpo_rm_parser.add_argument("--num-samples", type=int, default=100,
                             help="Number of preference pairs to generate")
    dpo_rm_parser.add_argument("--beta", type=float, default=0.1,
                             help="Temperature parameter for DPO loss")
    dpo_rm_parser.add_argument("--output-dir", type=str, default=None,
                             help="Directory to save the model")
    dpo_rm_parser.add_argument("--batch-size", type=int, default=4,
                             help="Batch size for training")
    dpo_rm_parser.add_argument("--epochs", type=int, default=1,
                             help="Number of training epochs")
    dpo_rm_parser.add_argument("--device", type=str, default=None,
                             help="Device to run training on (e.g., 'cpu', 'cuda:0')")
    dpo_rm_parser.add_argument("--verbose", action="store_true",
                             help="Enable verbose logging")
    
    # List available models
    list_parser = subparsers.add_parser("list-models", help="List available reward models")
    
    return parser.parse_args(args)

def check_gpu_requirements(args: argparse.Namespace):
    """Check if GPU is available and warn if not available but potentially needed.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    if not is_gpu_available() and args.device != "cpu":
        if args.command == "ppo" and "llama" in args.model.lower():
            logger.warning("Warning: LLaMA models typically require a GPU. Training on CPU may be very slow.")
        elif args.command in ["dpo", "dpo-rm"] and args.batch_size > 2:
            logger.warning("Warning: DPO training might be slow without a GPU. Consider reducing batch size.")

def run_ppo_command(args: argparse.Namespace):
    """Run PPO training with the given arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    # Parse lambda values if provided
    lambda_values = None
    if args.lambda_values:
        try:
            lambda_values = [float(x) for x in args.lambda_values.split(",")]
        except ValueError:
            logger.error("Invalid lambda values format. Should be comma-separated floats.")
            sys.exit(1)
    
    # Parse reward models
    reward_models = args.reward_models
    
    # Run training
    save_path = train_ppo(
        model_name=args.model,
        reward_models=reward_models,
        lambda_values=lambda_values,
        dataset_name=args.dataset,
        save_path=args.save_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        epochs=args.epochs,
        device=args.device,
        verbose=args.verbose
    )
    
    logger.info(f"PPO training completed. Model saved to {save_path}")

def run_dpo_command(args: argparse.Namespace):
    """Run DPO training with the given arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    # Run training
    output_dir = train_dpo(
        model_name=args.model,
        dataset_name=args.dataset,
        beta=args.beta,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device,
        verbose=args.verbose
    )
    
    logger.info(f"DPO training completed. Model saved to {output_dir}")

def run_dpo_rm_command(args: argparse.Namespace):
    """Run DPO training with a reward model.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    # Run training
    output_dir = train_dpo_with_reward_model(
        model_name=args.model,
        reward_model_name=args.reward_model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        beta=args.beta,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        verbose=args.verbose
    )
    
    logger.info(f"DPO training with reward model completed. Model saved to {output_dir}")

def list_models_command():
    """List available reward models."""
    reward_models = list_available_reward_models()
    print("Available reward models:")
    for model in reward_models:
        print(f"  - {model}")

def main(args: Optional[List[str]] = None):
    """Main entry point for the training CLI.
    
    Args:
        args (Optional[List[str]]): Command line arguments
    """
    parsed_args = parse_args(args)
    setup_logging(parsed_args.verbose if hasattr(parsed_args, 'verbose') else False)
    
    if parsed_args.command is None:
        logger.error("No command specified. Use -h for help.")
        sys.exit(1)
    
    # Check GPU requirements
    if parsed_args.command in ["ppo", "dpo", "dpo-rm"]:
        check_gpu_requirements(parsed_args)
    
    # Run appropriate command
    if parsed_args.command == "ppo":
        run_ppo_command(parsed_args)
    elif parsed_args.command == "dpo":
        run_dpo_command(parsed_args)
    elif parsed_args.command == "dpo-rm":
        run_dpo_rm_command(parsed_args)
    elif parsed_args.command == "list-models":
        list_models_command()
    else:
        logger.error(f"Unknown command: {parsed_args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 