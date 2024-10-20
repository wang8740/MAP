import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from JSON file
with open('plot_sequential_baseline_numeric_results.json', 'r') as f:
    data = json.load(f)

# Specify the model setting to use
model_setting = "opt1.3b-Anthropic-harmless"

# Extract data for the specified model setting
categories = data["model_settings"][model_setting]["categories"]
map_results = data["model_settings"][model_setting]["map_results"]
round_1_results = data["model_settings"][model_setting]["round_1_results"]
round_5_results = data["model_settings"][model_setting]["round_5_results"]


def plot_separate_setup():

    # Positions and width for the bars
    x = np.arange(len(categories))
    width = 0.25
    s = "70%" #50, 60, 70

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width, map_results[s], width, label='MAP', color='black')
    rects2 = ax.bar(x, round_1_results[s], width, label='Seq Round 1', color='blue')
    rects3 = ax.bar(x + width, round_5_results[s], width, label='Seq Round 5', color='cyan')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Aligned c-levels (in quantile)')
    ax.set_title('Comparison of Simultaneous (MAP) and Sequential Alignments')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Adding a horizontal line at y=0 for better visualization of positive and negative values
    ax.axhline(0, color='gray', linewidth=0.8)

    # Save the plot to a PDF
    plt.savefig(f'results/plot_sequential_baseline_{s}.pdf')


def plot_all_setups():

    # Define the s-values
    s_values = ["50%", "60%", "70%"]

    # Positions and width for the bars
    n_categories = len(categories)
    n_s_values = len(s_values)
    width = 0.2
    spacing = 0.1  # Spacing between different s-values within the same category
    group_spacing = 0.5  # Spacing between different categories

    # Compute new x positions for the bars
    x = np.arange(n_categories) * (n_s_values * 3 * width + (n_s_values - 1) * spacing + group_spacing)

    # Font size for all text elements
    font_size = 14

    # Beautiful colors for Nature journal
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 5))  # Adjusted height

    for i, s in enumerate(s_values):
        offset = i * (3 * width + spacing)
        for j, category in enumerate(categories):
            rects1 = ax.bar(x[j] + offset - width, map_results[s][j], width, label='MAP' if j == 0 and i == 0 else "", color=colors[0], alpha=0.8)
            rects2 = ax.bar(x[j] + offset, round_1_results[s][j], width, label='Seq Round 1' if j == 0 and i == 0 else "", color=colors[1], alpha=0.8)
            rects3 = ax.bar(x[j] + offset + width, round_5_results[s][j], width, label='Seq Round 5' if j == 0 and i == 0 else "", color=colors[2], alpha=0.8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Realized value levels (in quantile)', fontsize=font_size)
    ax.set_title('Comparison of Simultaneous (MAP) and Sequential Alignments', fontsize=font_size + 2)
    ax.set_ylim(0.3, ax.get_ylim()[1])  # Set y-axis limit

    # Create custom x-axis ticks
    outer_ticks = []
    inner_labels = []
    outer_labels = []
    for j, category in enumerate(categories):
        for i, s in enumerate(s_values):
            outer_ticks.append(x[j] + i * (3 * width + spacing) + width / 2)
            inner_labels.append(f'{s}')
        outer_labels.append(x[j] + ((n_s_values - 1) * (3 * width + spacing) / 2))

    # Adjust x-axis
    ax.set_xticks(outer_ticks)
    ax.set_xticklabels(inner_labels, fontsize=font_size)
    ax.set_xticks(outer_labels, minor=True)
    ax.set_xticklabels(categories, minor=True, fontsize=font_size)
    ax.tick_params(axis='x', which='minor', length=0, pad=25)

    # Create custom legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[0], alpha=0.8), 
            plt.Rectangle((0, 0), 1, 1, color=colors[1], alpha=0.8), 
            plt.Rectangle((0, 0), 1, 1, color=colors[2], alpha=0.8)]
    labels = ['MAP', 'Sequential Round 1', 'Sequential Round 5']
    ax.legend(handles, labels, ncol=3, fontsize=font_size)

    # Adding a horizontal line at y=0 for better visualization of positive and negative values
    ax.axhline(0, color='gray', linewidth=0.8)

    # Remove top and right spines (borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'results/fig_sequential_baseline.pdf', bbox_inches='tight')


if __name__ == "__main__":
    # plot_separate_setup()
    plot_all_setups()