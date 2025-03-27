"""Command-line interface for running model alignment."""

import os
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any

from alignmap.core.alignment import align_with_reward_models
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
        description="Run alignment with reward models"
    )
    
    # Add subparsers for different alignment methods
    subparsers = parser.add_subparsers(dest="command", help="Alignment method")
    
    # Basic alignment with reward models
    align_parser = subparsers.add_parser("align", help="Align with reward models")
    align_parser.add_argument("--model", type=str, required=True, 
                             help="Model name or path (e.g., 'gpt2', 'facebook/opt-1.3b')")
    align_parser.add_argument("--prompt", type=str, required=True,
                             help="Prompt to generate completions for")
    align_parser.add_argument("--reward-models", type=str, required=True,
                             help="Comma-separated list of reward model names, or 'all'")
    align_parser.add_argument("--lambda-values", type=str, default=None,
                             help="Comma-separated list of lambda values for reward models")
    align_parser.add_argument("--num-samples", type=int, default=16,
                             help="Number of completions to generate")
    align_parser.add_argument("--max-tokens", type=int, default=50,
                             help="Maximum number of tokens to generate")
    align_parser.add_argument("--temperature", type=float, default=1.0,
                             help="Sampling temperature")
    align_parser.add_argument("--device", type=str, default=None,
                             help="Device to run on (e.g., 'cpu', 'cuda:0')")
    align_parser.add_argument("--verbose", action="store_true",
                             help="Enable verbose logging")
    
    # Interactive alignment session
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive alignment session")
    interactive_parser.add_argument("--model", type=str, required=True,
                                  help="Model name or path (e.g., 'gpt2', 'facebook/opt-1.3b')")
    interactive_parser.add_argument("--reward-models", type=str, required=True,
                                  help="Comma-separated list of reward model names, or 'all'")
    interactive_parser.add_argument("--lambda-values", type=str, default=None,
                                  help="Comma-separated list of lambda values for reward models")
    interactive_parser.add_argument("--max-tokens", type=int, default=150,
                                  help="Maximum number of tokens to generate")
    interactive_parser.add_argument("--temperature", type=float, default=0.7,
                                  help="Sampling temperature")
    interactive_parser.add_argument("--device", type=str, default=None,
                                  help="Device to run on (e.g., 'cpu', 'cuda:0')")
    
    # List available models
    list_parser = subparsers.add_parser("list-models", help="List available reward models")
    
    return parser.parse_args(args)

def run_alignment(args: argparse.Namespace):
    """Run alignment with the given arguments.
    
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
    if args.reward_models == "all":
        reward_models = "all"
    else:
        reward_models = args.reward_models.split(",")
    
    # Run alignment
    result = align_with_reward_models(
        model_name=args.model,
        prompt=args.prompt,
        reward_models=reward_models,
        lambda_values=lambda_values,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        verbose=args.verbose
    )
    
    # Print results
    print("\n=== Alignment Results ===")
    print(f"Prompt: {args.prompt}")
    print(f"Best completion (reward: {result['best_reward']:.4f}):")
    print(f"{result['best_output']}")
    
    if args.verbose:
        print("\nAll completions:")
        for i, (completion, reward) in enumerate(zip(result['outputs'], result['rewards'])):
            print(f"\n{i+1}. Reward: {reward:.4f}")
            print(f"{completion}")
    
    print("\nReward model weights:")
    for model, weight in zip(result['reward_model_names'], result['lambda_values']):
        print(f"  {model}: {weight:.4f}")

def run_interactive_session(args: argparse.Namespace):
    """Run an interactive alignment session.
    
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
    if args.reward_models == "all":
        reward_models = "all"
    else:
        reward_models = args.reward_models.split(",")
    
    # Print welcome message
    print("\n=== Interactive Alignment Session ===")
    print("Type your prompts and press Enter to generate aligned completions.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the session.\n")
    
    # Start interactive loop
    try:
        while True:
            # Get user input
            prompt = input("Enter prompt: ")
            if prompt.lower() in ["exit", "quit"]:
                break
            
            if not prompt.strip():
                continue
            
            # Run alignment
            result = align_with_reward_models(
                model_name=args.model,
                prompt=prompt,
                reward_models=reward_models,
                lambda_values=lambda_values,
                num_samples=8,  # Fewer samples for interactive mode
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=args.device,
                verbose=False
            )
            
            # Print result
            print("\nAligned completion:")
            print(f"{result['best_output']}")
            print(f"(Reward: {result['best_reward']:.4f})")
            
    except KeyboardInterrupt:
        print("\nExiting interactive session.")

def list_models_command():
    """List available reward models."""
    reward_models = list_available_reward_models()
    print("Available reward models:")
    for model in reward_models:
        print(f"  - {model}")

def main(args: Optional[List[str]] = None):
    """Main entry point for the alignment CLI.
    
    Args:
        args (Optional[List[str]]): Command line arguments
    """
    parsed_args = parse_args(args)
    setup_logging(parsed_args.verbose if hasattr(parsed_args, 'verbose') else False)
    
    if parsed_args.command is None:
        logger.error("No command specified. Use -h for help.")
        sys.exit(1)
    
    # Run appropriate command
    if parsed_args.command == "align":
        run_alignment(parsed_args)
    elif parsed_args.command == "interactive":
        run_interactive_session(parsed_args)
    elif parsed_args.command == "list-models":
        list_models_command()
    else:
        logger.error(f"Unknown command: {parsed_args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 