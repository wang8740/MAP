import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
from genDPOdata import gen_mixed_preference_data
import wandb
import torch


def train_dpo(sample_size: int, beta: float, harmless_ratio: float, save_path, use_lora: bool = False):
    """Trains a Direct Preference Optimization (DPO) model with optional LoRA (Low-Rank Adaptation).

    Args:
        sample_size (int): Number of samples to use for training.
        beta (float): Regularization parameter for DPO training.
        harmless_ratio (float): Proportion of harmless to helpful data in the dataset.
        save_path (str): Directory to save the trained model.
        use_lora (bool): If True, use LoRA for fine-tuning a smaller, efficient model. Defaults to False.
    """

    # Initialize Weights and Biases
    wandb.init(
        project="value",  # Project name
        name="dpo_run",             # Optional: name for the specific run
        config={                         # Optional: log your configuration/hyperparameters
            "model_name": "facebook/opt-1.3b",
            "sample_size": sample_size,
            "beta": beta,
            "harmless_ratio": harmless_ratio,
            "use_lora": use_lora,
            "num_train_epochs": 1,
            "batch_size": 20,
        }
    )

    # Step 1: Load the tokenizer and model
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_lora:
        # Load the LoRA adapter if use_lora is True
        model = PeftModel.from_pretrained(
            model,
            "dpoModels_peft",
            is_trainable=True,
        )

    # Move the model to GPU
    model = model.to('cuda')

    data_source = {"harmless": harmless_ratio, "helpful": 1 - harmless_ratio}
    
    # Prepare datasets
    train_dataset = gen_mixed_preference_data(data_source, sample_size, split="train")
    # test_dataset = gen_mixed_preference_data(data_source, sample_size, split="test")

    # Initialize the DPO Trainer
    training_args = DPOConfig(
        beta=beta,
        output_dir=save_path,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=10,  # effective batch_size=20
        save_only_model=True,  # Don't save optimizer and scheduler state
    )
    
    if use_lora:
        # Pass the LoRA configuration if using LoRA
        dpo_trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
            tokenizer=tokenizer,
            peft_config=model.peft_config,  # Required for LoRA training
        )
    else:
        # No need to pass peft_config if training the full model
        dpo_trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
            tokenizer=tokenizer,
        )

    # Utility function to log GPU memory usage
    def log_gpu_memory_usage():
        """Logs the GPU memory usage
        
        This function tracks allocated, reserved, and free GPU memory, 
        logging these values to WandB for real-time monitoring during training.
        """
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
        free_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) - reserved
        print(f"\nAllocated Memory: {allocated:.2f} GB, Reserved Memory: {reserved:.2f} GB, Free Memory: {free_memory:.2f} GB")
        wandb.log({"gpu_memory_allocated_gb": allocated, "gpu_memory_reserved_gb": reserved, "gpu_free_memory_gb": free_memory})

    # Modify the compute_loss function to include GPU memory logging
    original_compute_loss = dpo_trainer.compute_loss

    def compute_loss_with_logging(*args, **kwargs):
        """Computes the loss while also logging GPU memory usage and loss values to WandB.

        Args:
            *args: Positional arguments for the compute_loss function.
            **kwargs: Keyword arguments for the compute_loss function.

        Returns:
            torch.Tensor: The computed loss value.
        """
        loss = original_compute_loss(*args, **kwargs)
        log_gpu_memory_usage()  # Log memory after every loss computation
        wandb.log({"train_loss": loss.item()})  # Log the loss value as well
        return loss
    
    dpo_trainer.compute_loss = compute_loss_with_logging

    # Train the model
    dpo_trainer.train()

    # Save the model
    dpo_trainer.save_model()

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    fire.Fire(train_dpo)

# Example usage: python trainDPO.py --harmless_ratio=0.5 --sample_size=2000
