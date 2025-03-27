import gradio as gr
import torch
import numpy as np
import json
from alignValues import AlignValues
import sys

def retrieve_rewards_min_max_avg(rewards, values_to_align):
    """
    Calculate min, max, and average for each value in the dataset using the rewards tensor.

    Args:
        rewards (torch.Tensor): A tensor of shape (k, n), where k is the number of values 
                                and n is the number of samples.
        values_to_align (list): List of value names corresponding to the rows in rewards.

    Returns:
        dict: A dictionary where each key is a value name, and the value is a dictionary
              with keys 'min', 'max', and 'avg', representing the statistics for that value.
    """
    rewards_np = rewards.numpy()
    stats = {}
    for idx, value in enumerate(values_to_align):
        stats[value] = {
            'min': float(np.min(rewards_np[idx])),
            'max': float(np.max(rewards_np[idx])),
            'avg': float(np.mean(rewards_np[idx])),
        }
    return stats


def estimate_realized_levels(lam, rewards):
    """
    Estimate the realized levels for all values using the given lambda.

    Args:
        lam (list or numpy.ndarray): A vector of weights with size matching the number of rows in rewards.
        rewards (torch.Tensor): A tensor of shape (k, n), where k is the number of values and n is the number of samples.

    Returns:
        torch.Tensor: A tensor of realized levels (size `k`) as the weighted sum for each value.
    """
    # Convert lambda to a tensor and ensure dimensions match
    lam_tensor = torch.tensor(lam, dtype=torch.float32)
    assert lam_tensor.size(0) == rewards.size(0), "Lambda dimension does not match the number of values."

    # Compute weights for each sample based on the softmax of the weighted sum of rewards
    weights = torch.softmax(torch.sum(lam_tensor[:, None] * rewards, dim=0), dim=0)

    # Compute realized levels for each value as a weighted sum across samples
    realized_levels = torch.matmul(rewards, weights)  # Shape (k,)
    return realized_levels


# Example test function
def test_example():
    # file_path = "results/opt1.3b-Anthropic-harmless.json"
    file_path = "results/llama2_chat-Anthropic-harmless.json"
    values_to_align = ["humor", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"]
    aligner = AlignValues(value_list=values_to_align, file_path=file_path)

    # Retrieve min, max, avg for each value
    stats = retrieve_rewards_min_max_avg(aligner.rewards, values_to_align)
    print("Statistics for rewards:")
    for value, stat in stats.items():
        print(f"{value}: min={stat['min']:.3f}, max={stat['max']:.3f}, avg={stat['avg']:.3f}")

    # Estimate realized value
    # lam = [0.5, 0.2, 0.3, 0.1, 0.1, 0.4]
    lam = [0, 0, 0, 0, 0, 0]
    realized_levels = estimate_realized_levels(lam, aligner.rewards)
    print(f"Estimated realized value: {realized_levels}")

test_example()
sys.exit('debug')

# Preload data outside the main function
values_to_align = ["humor", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"]
values_to_align_names = ["Humor", "Helpfulness", "Harmlessness", "Diversity", "Coherence", "Perplexity"]
file_paths = {
    "Llama2-7B": "results/llama2_chat-Anthropic-harmless.json",
    "OPT-1.3B": "results/opt1.3b-Anthropic-harmless.json",
}

# Initialize AlignValues instances for both models
aligners = {
    model: AlignValues(values_to_align, file_path)
    for model, file_path in file_paths.items()
}

# Generate slider settings from preloaded data
slider_stats = {
    model: retrieve_rewards_min_max_avg(aligner.rewards, values_to_align)
    for model, aligner in aligners.items()
}

# After initializing aligners
print("Debug: Aligners initialized")
for model, aligner in aligners.items():
    print(f"Debug: {model} aligner rewards shape: {aligner.rewards.shape}")

# After generating slider settings
print("Debug: Slider settings generated")
for model, stats in slider_stats.items():
    print(f"Debug: {model} stats: \n{stats}")


def main(model_choice, *c_values):
    print(f"Debug: Entering main with model_choice = {model_choice}, c_values = {c_values}")
    
    if model_choice is None:
        return "Please select a model", [0] * len(c_values)  # Default values if no model is chosen

    aligner = aligners[model_choice]
    aligner.c = torch.tensor(c_values, dtype=torch.float32)  # Update palette
    print(f"Debug: aligner.c set to {aligner.c.tolist()}")

    lam, success = aligner.optimize_lambda(verbose=False)
    print(f"Debug: optimize_lambda returned lam = {lam}, success = {success}")

    if success:
        realized_levels = estimate_realized_levels(lam, aligner.rewards)
        realized_levels = [float(level) for level in realized_levels]
        return f"Optimization successful! Lambda values: {lam}", realized_levels
    else:
        c_low = [stat["avg"] for stat in slider_stats[model_choice].values()]
        c_high = c_values
        print(f"Debug: c_low = {c_low}, c_high = {c_high}")

        adjust_success, adjust_c, lam = aligner.find_pareto_by_interpolation(c_low, c_high)

        if adjust_success:
            realized_levels = estimate_realized_levels(lam, aligner.rewards)
            realized_levels = [float(level) for level in realized_levels]
            return f"Your specified palette is infeasible. Adjusted to feasible palette using c = {adjust_c}.", realized_levels
        else:
            print("Debug: Auto-adjustment failed.")
            return "Something wrong with the auto-adjustment.", c_values

def create_sliders(model):
    return [
        gr.Slider(
            minimum=stat["min"],
            maximum=stat["max"],
            value=stat["avg"],
            step=0.1,
            label=fr"c_{i + 1}: {values_to_align_names[i]}",
            interactive=True,
            show_label=False  # Remove individual clear buttons
        ) for i, stat in enumerate(slider_stats[model].values())
    ]

def update_sliders(model):
    if model is None:
        return [0] * len(values_to_align)  # Default values if no model is selected
    
    return [
        gr.Slider.update(
            minimum=stat["min"],
            maximum=stat["max"],
            value=stat["avg"],
            label=fr"c_{i + 1}: {values_to_align_names[i]}"
        ) for i, stat in enumerate(slider_stats[model].values())
    ]

with gr.Blocks() as iface:
    model_choice = gr.Dropdown(
        choices=["Llama2-7B", "OPT-1.3B"], 
        value="Llama2-7B", 
        label="Choose Model"
    )
    
    with gr.Row():
        with gr.Column():
            gr.Label("Value Palette")
            input_sliders = create_sliders("Llama2-7B")  # Initial sliders for default model
            
        with gr.Column():
            gr.Label("Realized Value Levels")
            output_sliders = create_sliders("Llama2-7B")  # Output sliders initialized
            for slider in output_sliders:
                slider.interactive = False
    
    result_box = gr.Textbox(label="Optimization Results")
    reset_btn = gr.Button("Reset")

    def reset_to_avg(model_choice):
        return update_sliders(model_choice)

    # Set up event handlers
    inputs = [model_choice] + input_sliders
    outputs = [result_box] + output_sliders

    model_choice.change(
        fn=lambda model: update_sliders(model),
        inputs=model_choice,
        outputs=input_sliders
    )

    reset_btn.click(
        fn=reset_to_avg,
        inputs=[model_choice],
        outputs=input_sliders
    )


if __name__ == '__main__':
    _, _, url = iface.queue().launch(server_name="0.0.0.0", share=False)
    print("Public URL:", url)
