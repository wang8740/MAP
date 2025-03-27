import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import ALL_SUPPORTED_VALUES, ALL_SUPPORTED_VALUES_plotnames
import numpy as np
import torch
import matplotlib.patches as mpatches

def plot_hist(json_file: str) -> None:
    """Generate histograms for each reward type in a JSON file, visualizing reward distributions.

    This function reads a JSON file containing calculated rewards for various sentences, converts
    the data into a DataFrame, and plots histograms for each reward category. Each histogram represents
    the frequency distribution of rewards for a specific category.

    Args:
        json_file (str): Path to the JSON file containing reward data.

    Example:
        >>> json_file = "results/opt1.3b-Anthropic-harmless.json"
        >>> plot_hist(json_file)
    """

    # Step 1: Read the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Step 2: Prepare the data
    # Convert the list of dictionaries into a DataFrame for easier processing
    df = pd.DataFrame(data)

    # Step 3: Create subplots for each key
    # Number of rows needed for subplots based on the number of keys
    rows = len(ALL_SUPPORTED_VALUES)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 5 * rows))

    # Plot a histogram on each subplot
    for i, key in enumerate(ALL_SUPPORTED_VALUES):
        axes[i].hist(df[key], bins=20, edgecolor='black')
        axes[i].set_title(f'Histogram of {key}')
        axes[i].set_xlabel(f'Values of {key}')
        axes[i].set_ylabel('Frequency')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Step 4: Save the plot to a PDF file
    pdf_path = json_file.replace('.json', '_reward_hist.pdf')
    plt.savefig(pdf_path, format='pdf')
    print(f"Plot saved to {pdf_path}")

    # Close the plot to free memory
    plt.close()


def plot_weighted_unweighted_histograms(
    file_path: str, 
    values_to_evaluate: list[str], 
    values_to_align: list[str], 
    lam: list[float], 
    subplot_names: list[str], 
    save_path: str
) -> None:
    """Plot weighted and unweighted histograms for reward distributions, showing the effects of MAP alignment.

    This function reads a JSON file with reward data, applies MAP alignment using specified lambda weights,
    and generates histograms for each reward type before and after alignment. The histograms allow comparison
    between original and MAP-aligned distributions for each reward type.

    Args:
        file_path (str): Path to the JSON file containing reward data.
        values_to_evaluate (list[str]): Reward types to evaluate and plot (e.g., ["humor", "gpt2-helpful"]).
        values_to_align (list[str]): Reward types to align based on the lambda weights.
        lam (list[float]): Lambda values used to adjust reward weights in alignment.
        subplot_names (list[str]): Names to use for subplots, corresponding to each value in `values_to_evaluate`.
        save_path (str): Path to save the resulting PDF file of histograms.

    Example:
        >>> file_path = "results/llama2_chat-Anthropic-harmless.json"
        >>> values_to_evaluate = ["humor", "gpt2-helpful", "gpt2-harmless"]
        >>> values_to_align = ["humor", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"]
        >>> lam = [5.942, 2.432, 2.923, 0.006, 0.011, 0.147]
        >>> subplot_names = ["Humor", "Helpfulness", "Harmlessness"]
        >>> save_path = "results/fig_hist_llama2chat_80.pdf"
        >>> plot_weighted_unweighted_histograms(file_path, values_to_evaluate, values_to_align, lam, subplot_names, save_path)
    """
    # Convert lambda to tensor
    lam_tensor = torch.tensor(lam, dtype=torch.float32)
    
    # Load data
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Prepare figures
    num_values = len(values_to_evaluate)
    fig, axes = plt.subplots(nrows=(num_values + 2) // 3, ncols=3, figsize=(18, 6))  # Adjust for 3 plots per row
    axes = axes.flatten()

    # Process each value to evaluate
    for idx, value_to_evaluate in enumerate(values_to_evaluate):
        # Prepare data for weights calculation
        rewards = []
        for v in values_to_align:
            value_rewards = [entry[v] for entry in data]
            tensor_rewards = torch.tensor(value_rewards, dtype=torch.float32)
            rewards.append(tensor_rewards)
        rewards = torch.stack(rewards, dim=0)

        # Compute the softmax of lambda times rewards
        weights = torch.softmax(torch.sum(lam_tensor[:, None] * rewards, dim=0), dim=0)

        # Gather the rewards of the value to evaluate
        rewards_of_value_to_eval = [entry[value_to_evaluate] for entry in data]
        tensor_rewards_of_value_to_eval = torch.tensor(rewards_of_value_to_eval, dtype=torch.float32)
        updated_c = torch.sum(weights * tensor_rewards_of_value_to_eval).item()
        original_c = np.mean(rewards_of_value_to_eval)

        # Convert to DataFrame for Seaborn compatibility
        df = pd.DataFrame({
            'Rewards': tensor_rewards_of_value_to_eval.numpy(),
            'Weights': weights.numpy()
        })

        # Plotting histograms using seaborn for better visualization
        ax = axes[idx]
        sns.histplot(df, x='Rewards', bins=30, ax=ax, color="blue", kde=False, stat="density", alpha=0.2, edgecolor=None, label='Distribution of reward scores under original model')
        sns.histplot(df, x='Rewards', weights='Weights', bins=30, ax=ax, color="green", kde=False, stat="density", alpha=0.2, edgecolor=None, label='Distribution of reward scores under MAP-aligned model')

        # Add KDE plots with filled area
        sns.kdeplot(data=df, x='Rewards', weights='Weights', ax=ax, color="green", bw_adjust=0.5, fill=True, linewidth=1, alpha=0.2)
        sns.kdeplot(df['Rewards'], ax=ax, color="blue", bw_adjust=0.5, fill=True, linewidth=1, alpha=0.2)

        # Add vertical lines for averages
        ax.axvline(original_c, color='blue', linestyle='-', linewidth=3, label='Original realized value levels')
        ax.axvline(updated_c, color='green', linestyle='--', linewidth=3, label='MAP realized value levels')

        # Calculate the maximum y-value for setting the arrow height
        max_y_value = ax.get_ylim()[1] * 0.5  # xxx% of the way up the y-axis
        # Add an arrow with higher positioning
        ax.annotate('', xy=(updated_c, max_y_value), xytext=(original_c, max_y_value),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))

        # Set x-axis limits based on data percentiles to avoid extreme values
        lower_bound, upper_bound = np.percentile(rewards_of_value_to_eval, [0, 100])
        ax.set_xlim(lower_bound, upper_bound)

        # ax.set_title(f"{value_to_evaluate}")
        ax.set_xlabel(f"{subplot_names[idx]} scores", fontsize=20)
        ax.set_ylabel('Density', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)

        if idx == 0:  # Collect legend handles only from the first subplot for no repetition
            handles, labels = ax.get_legend_handles_labels()

    # Create a single shared legend for all plots:
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True, fontsize=20, edgecolor='black', facecolor='white')

    # Adjust layout to make space for the legend on top
    plt.subplots_adjust(top=0.9)  # Adjust the top parameter to fit the legend above the plots

    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

    return
 

def plot_hist_positive(json_file: str) -> None:
    """Plot a histogram for the 'positive' column in a JSON file, showing value distribution.

    This function reads a JSON file with a 'positive' column, applies an exponential transformation to
    the values, and creates a histogram. It’s used to visualize the distribution of transformed 'positive'
    values across the dataset.

    Args:
        json_file (str): Path to the JSON file containing a 'positive' column.

    Example:
        >>> json_file = "results/opt1.3b-positive_values.json"
        >>> plot_hist_positive(json_file)
    """

    with open(json_file, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if 'positive' not in df.columns:
        print("Error: 'positive' column not found in the JSON data.")
        return

    plt.figure(figsize=(10, 5))
    plt.hist(np.exp(df['positive']), bins=20, edgecolor='black')
    plt.title('Histogram of Positive Values')
    plt.xlabel('Values of Positive')
    plt.ylabel('Frequency')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Step 4: Save the plot to a PDF file
    pdf_path = json_file.replace('.json', '_positive_hist.pdf')
    plt.savefig(pdf_path, format='pdf')
    print(f"Plot saved to {pdf_path}")

    # Close the plot to free memory
    plt.close()



if __name__ == '__main__':


    # opt1.3b 70% all
    # file_path = 'results/opt1.3b-Anthropic-harmless.json'
    # lam = [12.766,1.526,1.689,0.012,0.019,0.023]
    # save_path = 'results/fig_hist_opt.pdf'

    # llama2_chat 80% all
    file_path = 'results/llama2_chat-Anthropic-harmless.json'
    values_to_align = ["humor", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"]
    lam = [5.942,2.432,2.923,0.006,0.011,0.147]
    save_path = 'results/fig_hist_llama2chat_80.pdf'

    # values_to_align = ["humor"]
    # lam = [2.887]
    # save_path = 'results/fig_hist_llama2chat_humor80.pdf'

    values_to_evaluate = ["humor", "gpt2-helpful", "gpt2-harmless"]
    subplot_names = ["Humor", "Helpfulness", "Harmlessness"]

    plot_weighted_unweighted_histograms(file_path, values_to_evaluate, values_to_align, lam, subplot_names, save_path)

