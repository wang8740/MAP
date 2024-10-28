import json
import math
import os
import glob
import matplotlib.pyplot as plt
import itertools
import csv
import numpy as np

def calculate_win_rate(
    model_file: str, 
    base_model_file: str, 
    metrics: list[str] = ['perplexity', 'coherence', 'diversity', 'gpt2-harmless', 'gpt2-helpful', 'humor']
) -> dict:
    """Calculate win rates of model performance compared to a base model across specified metrics.

    Opens JSON files for the provided models, calculates the win rate for each metric,
    and the standard error of each win rate.

    Args:
        model_file (str): Path to the JSON file of the fine-tuned model's generated continuations.
        base_model_file (str): Path to the JSON file of the base model's generated continuations.
        metrics (list[str], optional): List of human values/metrics for comparison. Defaults to a standard list.

    Returns:
        dict: Contains file paths, win rates for each metric, and standard errors for each metric.

    Example:
        >>> result = calculate_win_rate("fine_tuned_model.json", "base_model.json")
        >>> print(result)

    Command-line usage:
        python script.py --model_file="fine_tuned_model.json" --base_model_file="base_model.json"
    """

    with open(model_file, 'r') as f:
        model_data = json.load(f)
    
    with open(base_model_file, 'r') as f:
        base_model_data = json.load(f)
    
    assert len(model_data) == len(base_model_data), f"Mismatched number of entries in file {model_file}."
    
    win_counts = {metric: 0 for metric in metrics}
    
    for model_item, base_model_item in zip(model_data, base_model_data):
        for metric in metrics:
            if model_item[metric] > base_model_item[metric]:
                win_counts[metric] += 1
    
    total_entries = len(model_data)
    win_rates = {metric: win_counts[metric] / total_entries for metric in metrics}
    win_rate_se = {f"{metric}_SE": math.sqrt(win_rates[metric] * (1 - win_rates[metric]) / total_entries) for metric in metrics}
    
    win_rates = {metric: f"{win_rates[metric]:.2f}" for metric in win_rates}
    win_rate_se = {metric: f"{se:.2f}" for metric, se in win_rate_se.items()}
    
    result = {
        'model-path': model_file,
        'basemodel-path': base_model_file,
        **win_rates,
        **win_rate_se
    }
    
    return result


def collect_multiple_results(
    model_files: list[str], 
    base_model_file: str, 
    file_prefix: str, 
    metrics: list[str] = None
) -> list[dict]:
    """Aggregate win rate results for multiple models compared to a base model and save to a JSON file.

    Iterates over multiple model files, calculates win rates for each using `calculate_win_rate`, 
    and saves the aggregate results as JSON.

    Args:
        model_files (list[str]): List of file paths for the fine-tuned model JSON files.
        base_model_file (str): Path to the JSON file for the base model.
        file_prefix (str): Prefix for the output JSON file name.
        metrics (list[str], optional): Metrics for win rate calculation. Defaults to None.

    Returns:
        list[dict]: List of win rate results for each model file.

    Example:
        >>> base_model_file = 'results/opt1.3b-Anthropic-harmless.json'
        >>> harmless_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> beta = 0.1
        >>> file_prefix = f"results_comparison/winrate_{beta}beta_DPO"
        >>> model_files = [f'modelsDPO/opt1.3b-2000sample-{beta}beta-{ratio}harmless-Anthropic-harmless.json' for ratio in harmless_ratios]
        >>> win_rate_results_list = collect_multiple_results(model_files, base_model_file, file_prefix)
        >>> print(win_rate_results_list)
    
        It will also create a file file_prefix.json that contains a list of entries like this:
        [
            {
                "model-path": "modelsDPO/soup/opt1.3b-2000sample-0.5beta-0.1soup-Anthropic-harmless.json",
                "basemodel-path": "results/opt1.3b-Anthropic-harmless.json",
                "perplexity": "0.70",
                "coherence": "0.48",
                "diversity": "0.48",
                "gpt2-harmless": "0.62",
                "gpt2-helpful": "0.54",
                "humor": "0.21",
                "perplexity_SE": "0.01",
                "coherence_SE": "0.01",
                "diversity_SE": "0.01",
                "gpt2-harmless_SE": "0.01",
                "gpt2-helpful_SE": "0.01",
                "humor_SE": "0.01"
            },
            ...
        ]
    """

    # List to store all results
    win_rate_results_list = []

    # Loop through each harmless_ratio and calculate win rates
    for model_file in model_files:
        if os.path.exists(model_file):
            win_rate_result = calculate_win_rate(model_file, base_model_file, metrics)
            win_rate_results_list.append(win_rate_result)
        else:
            print(f"Model file {model_file} not found.")

    # Save win_rate_results_list to JSON file
    with open(f"{file_prefix}.json", 'w') as f:
        json.dump(win_rate_results_list, f, indent=4)
    print(f"saved results to {file_prefix}.json")

    return win_rate_results_list


def render_latex_table(win_rate_results_list: list[dict], file_prefix: str) -> str:
    """Generate and save a LaTeX table for win rates from a list of results.

    Constructs a LaTeX table summarizing win rates across models and metrics.
    Saves the table to a .tex file for LaTeX compilation.

    Args:
        win_rate_results_list (list[dict]): List of dictionaries containing win rate results.
        file_prefix (str): Prefix for the output LaTeX file name.

    Returns:
        str: LaTeX-formatted table as a string.
    """

    metrics = ['perplexity', 'coherence', 'diversity', 'gpt2-harmless', 'gpt2-helpful', 'humor']
    
    # Begin LaTeX table with booktabs package
    latex_table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{l" + "c" * len(metrics) + "}\n\\toprule\n"
    
    # Header row
    latex_table += "Model & " + " & ".join(metrics) + " \\\\\n\\midrule\n"
    
    # Data rows
    for result in win_rate_results_list:
        model_name = os.path.basename(result['model-path'])
        row = [f"{model_name}"] + [result[metric] for metric in metrics]
        latex_table += " & ".join(row) + " \\\\\n"
    
    latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{Win rates for different models and metrics}\n\\end{table}"

    with open(f"{file_prefix}.tex", 'w') as f:
        f.write(latex_table)
    print(f"saved results to {file_prefix}.tex")

    return latex_table


def plot_helpful_vs_harmless(win_rate_results_list: list[dict], harmless_ratios: list[float], file_prefix: str) -> None:
    """Plot and save a line graph of helpful and harmless win rates vs. harmless ratios.

    Creates a plot comparing helpfulness and harmlessness win rates as a function of
    different harmless ratios. Saves the plot as a PDF file.

    Args:
        win_rate_results_list (list[dict]): List of dictionaries containing win rate results.
        harmless_ratios (list[float]): List of harmlessness ratio values to plot.
        file_prefix (str): Prefix for the output PDF file name.

    """
    helpful = [float(result['gpt2-helpful']) for result in win_rate_results_list]
    harmless = [float(result['gpt2-harmless']) for result in win_rate_results_list]
    
    plt.figure(figsize=(8, 5))
    plt.plot(harmless_ratios, helpful, marker='o', label='gpt2-helpful', color='blue')
    plt.plot(harmless_ratios, harmless, marker='s', label='gpt2-harmless', color='green')
    plt.xlabel('Harmless Ratio')
    plt.ylabel('Win Rate')
    plt.title('Win Rate of Helpful and Harmless vs Harmless Ratio')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig(f"{file_prefix}.pdf", bbox_inches='tight')  # bbox_inches='tight' removes extra whitespace
    plt.close()
    print(f"saved results to {file_prefix}.pdf")


def plot_winrate() -> None:
    """Generate a scatter plot comparing win rates for various models based on helpfulness and harmlessness.

    This function plots a 2D scatter plot where each model entry is represented as a point, 
    with the x-axis representing "gpt2-helpful" scores and the y-axis representing "gpt2-harmless" scores.
    Each baseline model is given a unique marker and color, and the function computes and displays
    navigation efficiency for each model (the proportion of points in the "upper right" quadrant).

    A reference point for the original model is plotted, along with gridlines and shading to highlight 
    the upper-right region, which represents favorable scores for both helpfulness and harmlessness.

    Specifically, we run __main__ to obtain the following baselines and our method (MAP) and their generated result files:

        "DPO(0.1)": results_comparison/winrate_0.1beta_DPO.json
        "DPO(0.5)": results_comparison/winrate_0.5beta_DPO.json
        "DPO-Soup(0.1)": results_comparison/winrate_0.1beta_DPOsoup.json
        "DPO-Soup(0.5)": results_comparison/winrate_0.5beta_DPOsoup.json
        r"MoRL with random $\lambda$": results_comparison/winrate_6scale_2valuesHH_PPO_DirichletRand.json
        r"MAP with feasible $\lambda$": results_comparison/winrate_6scale_2valuesHH_PPO_MapRand.json

    Each file contains a list of entries like this:
    [
        {
            "model-path": "modelsDPO/soup/opt1.3b-2000sample-0.5beta-0.1soup-Anthropic-harmless.json",
            "basemodel-path": "results/opt1.3b-Anthropic-harmless.json",
            "perplexity": "0.70",
            "coherence": "0.48",
            "diversity": "0.48",
            "gpt2-harmless": "0.62",
            "gpt2-helpful": "0.54",
            "humor": "0.21",
            "perplexity_SE": "0.01",
            "coherence_SE": "0.01",
            "diversity_SE": "0.01",
            "gpt2-harmless_SE": "0.01",
            "gpt2-helpful_SE": "0.01",
            "humor_SE": "0.01"
        },
        ...
    ]
    We can call plot_winrate() to plot a figure titled WinRate where each entry becomes a 2D point with x-axis "gpt2-helpful" and y-axis "gpt2-harmless"
    Each baseline name will get a different legend in the same figure. 

    Returns:
        None

    Example:
        >>> plot_winrate()

    """
    # Define file paths and baseline names
    baselines = {
        r"MAP-D (random $\mathbf{\lambda}$)": "results_comparison/winrate_6scale_2valuesHH_Decoding_MapRand.json",
        r"MAP-F (random $\mathbf{\lambda}$)": "results_comparison/winrate_6scale_2valuesHH_PPO_MapRand.json",
        r"MORL-D (random $\mathbf{\lambda}$)": "results_comparison/winrate_6scale_2valuesHH_Decoding_DirichletRand.json",
        r"MORL-F (random $\mathbf{\lambda}$)": "results_comparison/winrate_6scale_2valuesHH_PPO_DirichletRand.json",
        r"DPO($\beta=$0.1)": "results_comparison/winrate_0.1beta_DPO.json",
        r"DPO($\beta=$0.5)": "results_comparison/winrate_0.5beta_DPO.json",
        r"DPO-Soup($\beta=$0.1)": "results_comparison/winrate_0.1beta_DPOsoup.json",
        r"DPO-Soup($\beta=$0.5)": "results_comparison/winrate_0.5beta_DPOsoup.json",
    }

    # Initialize the plot
    plt.figure(figsize=(8, 6))

    # Color and marker iterators
    colors = itertools.cycle(['b', 'navy', 'orange', 'darkorange', 'm', 'k', 'g', 'c'])
    markers = itertools.cycle(['o', 'H', 'd', '*', 'P', 'X', 's', 'D'])
    Ksigma = 0.03

    # Function to calculate navigation efficiency
    def calculate_navigation_efficiency(data):
        count_total = len(data)
        count_upper_right = sum(1 for entry in data if float(entry["gpt2-helpful"]) >= 0.5-Ksigma and float(entry["gpt2-harmless"]) >= 0.5-Ksigma)
        return count_upper_right / count_total if count_total > 0 else 0

    # Add jitter function
    def add_jitter(values, jitter_amount=0.001):
        return values + np.random.uniform(-jitter_amount, jitter_amount, size=len(values))

    handles = []  # Store scatter plot handles for custom legend

    # Iterate over the baselines and plot the points
    for label, file_path in baselines.items():
        with open(file_path, 'r') as file:
            data = json.load(file)
            gpt2_helpful = [float(entry["gpt2-helpful"]) for entry in data]
            gpt2_harmless = [float(entry["gpt2-harmless"]) for entry in data]
            
            # Calculate and print navigation efficiency
            nav_efficiency = calculate_navigation_efficiency(data)
            print(f"Navigation efficiency for {label}: {nav_efficiency:.2%}")

            # Add jitter to the values
            gpt2_helpful_jittered = add_jitter(np.array(gpt2_helpful))
            gpt2_harmless_jittered = add_jitter(np.array(gpt2_harmless))
            
            # Plot each baseline as a scatter plot with unique color and marker
            color = next(colors)
            marker = next(markers)
            scatter_plot = plt.scatter(gpt2_helpful_jittered, gpt2_harmless_jittered, color=color, marker=marker, s=15, edgecolor='none', alpha=0.6)

            # Store the handle for creating a custom legend
            handles.append(scatter_plot)

            # Calculate and plot the average point for each baseline
            avg_helpful = sum(gpt2_helpful) / len(gpt2_helpful)
            avg_harmless = sum(gpt2_harmless) / len(gpt2_harmless)
            jitter_helpful = avg_helpful + np.random.normal(0, 0.01)
            jitter_harmless = avg_harmless + np.random.normal(0, 0.01)
            plt.scatter(jitter_helpful, jitter_harmless, color=color, marker=marker, s=120, alpha=0.9)

    # Plot a reference red circle at (0.5, 0.5)
    plt.scatter(0.5, 0.5, color='r', marker='o', s=120, label="Original model")

    # Add gridlines at 0.5 intervals to highlight the upper right regime
    # plt.axhline(y=0.5, color='gray', linestyle='--')
    # plt.axvline(x=0.5, color='gray', linestyle='--')

    # Define the band limits for horizontal and vertical reference lines
    y_band_lower = 0.5 - Ksigma
    y_band_upper = 0.5
    x_band_lower = 0.5 - Ksigma
    x_band_upper = 0.5
    ymin, ymax = -0.0, 1.0
    plt.fill_between(x=[0.5, 1], y1=y_band_lower, y2=y_band_upper, color='gray', alpha=0.2)
    plt.fill_betweenx(y=[0.5, 1], x1=x_band_lower, x2=x_band_upper, color='gray', alpha=0.2)
    # plt.axhline(y=y_band_upper, color='gray', linestyle='--', linewidth=2)
    # plt.axvline(x=x_band_upper, color='gray', linestyle='--', linewidth=2)
    plt.hlines(y=y_band_upper, xmin=0.5, xmax=1, color='gray', linestyle='--', linewidth=2)
    plt.vlines(x=x_band_upper, ymin=0.5, ymax=ymax, color='gray', linestyle='--', linewidth=2)

    # Set labels and ticks
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15) 
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlim(0, 1)
    plt.ylim(ymin, ymax)
    plt.xlabel("Helpfulness", fontsize=16)
    plt.ylabel("Harmlessness", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Win Rate against the Original Model", fontsize=16)

    # plt.legend(fontsize=14)

    # Create a custom legend with large marker size for legend only
    legend_elements = [plt.Line2D([0], [0], marker=h.get_paths()[0], color='w', markerfacecolor=h.get_facecolor()[0], 
                                  markersize=10, alpha=1, label=label) 
                       for h, label in zip(handles, baselines.keys())]

    # Add the base model marker to the legend
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label="Original model"))

    plt.legend(handles=legend_elements, fontsize=12, loc='lower left')

    # Save the figure
    plt.savefig("results_comparison/fig_compare_winrate.pdf", bbox_inches='tight')

    print("Plot successfully saved to fig_compare_winrate.pdf!")


def plot_cLevels():
    """Generate a scatter plot to compare average rewards (c-level) across various model baselines.

    This function visualizes the average reward levels (c-level) for multiple model baselines,
    using the "gpt2-helpful" metric as the x-axis and the "gpt2-harmless" metric as the y-axis.
    Each baseline has its own color and marker style for distinction. 
    A reference model, indicated by a red circle, is included at the original model's values.
    
    Baselines include DPO with various ratios, DPO-Soup, and MAP/MoRL with random or feasible lambda.
    Each CSV file from the models contains metrics, and this function extracts the "avg" row
    to plot the gpt2-helpful and gpt2-harmless values.

    Specifically, we make a plot that compares the average reward (c-level) using the following baselines (generated from __main__)
        "DPO(0.1)": 
            harmless_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for ratio in harmless_ratios:
                model_files.append(f'modelsDPO/opt1.3b-2000sample-0.1beta-{ratio}harmless-Anthropic-harmless.csv')

        "DPO(0.5)":
            for ratio in harmless_ratios:
                model_files.append(f'modelsDPO/opt1.3b-2000sample-0.5beta-{ratio}harmless-Anthropic-harmless.csv')

        "DPO-Soup(0.1)":
            for ratio in harmless_ratios:
                model_files.append(f'modelsDPO/soup/opt1.3b-2000sample-0.1beta-{ratio}soup-Anthropic-harmless.csv')

        "DPO-Soup(0.5)":
            for ratio in harmless_ratios:
                model_files.append(f'modelsDPO/soup/opt1.3b-2000sample-0.5beta-{ratio}soup-Anthropic-harmless.csv')

        r"MoRL with random $\lambda$":
            all csv files under modelsPPO/random-lambda/

        r"MAP with feasible $\lambda$ (Our proposed)":
            all csv files under modelsPPO/MAP-lambda

        Each csv file contains like this template:
            Statistic,humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
            avg,1.771,-1.509,0.315,0.871,0.39,-2.785
            avg_std,0.028,0.022,0.024,0.002,0.004,0.01
            50%,2.421,-1.576,0.42,0.906,0.402,-2.745
            60%,2.471,-1.319,0.725,0.918,0.455,-2.654
            70%,2.506,-1.036,1.05,0.928,0.51,-2.568
            80%,2.529,-0.672,1.357,0.937,0.566,-2.452
            90%,2.551,-0.127,1.722,0.945,0.641,-2.286
            99%,2.584,1.12,2.486,0.958,0.792,-1.801

        We only extract the two columns gpt2-helpful (x axis) and gpt2-harmless (y axis) under the first row "avg"
        and draw a plot.

    Args:
        None

    Returns:
        None: The plot is saved as a PDF file in `results_comparison/fig_compare_avg_reward.pdf`.

    Example:
        >>> plot_cLevels()

    """

    # Define the file paths and baseline names
    baselines = {
        r"MAP-D (random $\mathbf{\lambda}$)": [f'modelsDecoding/MAP-lambda/{file}' for file in os.listdir('modelsDecoding/MAP-lambda/') if file.endswith('.csv')],
        r"MAP-F (random $\mathbf{\lambda}$)": (
            [f'modelsPPO/MAP-lambda2/{file}' for file in os.listdir('modelsPPO/MAP-lambda2/') if file.endswith('.csv')] +
            [f'modelsPPO/MAP-lambda/{file}' for file in os.listdir('modelsPPO/MAP-lambda/') if file.endswith('.csv')]
        ),
        r"MORL-D (random $\mathbf{\lambda}$)": [f'modelsDecoding/random-lambda/{file}' for file in os.listdir('modelsDecoding/random-lambda/') if file.endswith('.csv')],
        r"MORL-F (random $\mathbf{\lambda}$)": (
            [f'modelsPPO/random-lambda2/{file}' for file in os.listdir('modelsPPO/random-lambda2/') if file.endswith('.csv')] +
            [f'modelsPPO/random-lambda/{file}' for file in os.listdir('modelsPPO/random-lambda/') if file.endswith('.csv')]
        ),
        r"DPO($\beta=$0.1)": [f'modelsDPO/opt1.3b-2000sample-0.1beta-{ratio}harmless-Anthropic-harmless_realC-levels.csv' for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
        r"DPO($\beta=$0.5)": [f'modelsDPO/opt1.3b-2000sample-0.5beta-{ratio}harmless-Anthropic-harmless_realC-levels.csv' for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
        r"DPO-Soup($\beta=$0.1)": [f'modelsDPO/soup/opt1.3b-2000sample-0.1beta-{ratio}soup-Anthropic-harmless_realC-levels.csv' for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
        r"DPO-Soup($\beta=$0.5)": [f'modelsDPO/soup/opt1.3b-2000sample-0.5beta-{ratio}soup-Anthropic-harmless_realC-levels.csv' for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
    }

    # Initialize the plot
    plt.figure(figsize=(8, 6))

    # Color and marker iterators
    colors = itertools.cycle(['b', 'navy', 'orange', 'darkorange', 'm', 'k', 'g', 'c'])
    markers = itertools.cycle(['o', 'H', 'd', '*', 'P', 'X', 's', 'D'])
    Ksigma = 0.06

    # Add jitter function
    def add_jitter(values, jitter_amount=0.001):
        return values + np.random.uniform(-jitter_amount, jitter_amount, size=len(values))

    handles = []  # Store scatter plot handles for custom legend

    # Function to extract the average gpt2-helpful and gpt2-harmless values from CSV files
    def extract_avg_values(file):
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'avg':  # Extract the row corresponding to "avg"
                    return float(row[2]), float(row[3])  # gpt2-helpful (column 2), gpt2-harmless (column 3)

    # Function to calculate navigation efficiency
    def calculate_navigation_efficiency(helpful_vals, harmless_vals, helpful_ref, harmless_ref):
        total_points = len(helpful_vals)
        points_above_ref = sum(1 for h, hh in zip(helpful_vals, harmless_vals) if h >= helpful_ref-Ksigma and hh >= harmless_ref-Ksigma)
        return points_above_ref / total_points if total_points > 0 else 0

    # Extract the reference values
    gpt2_helpful_ref, gpt2_harmless_ref = extract_avg_values('results/opt1.3b-Anthropic-harmless_realC-levels.csv')

    # Iterate over the baselines and plot the points
    for label, file_list in baselines.items():
        gpt2_helpful_vals, gpt2_harmless_vals = [], []

        # Extract values for each file in the baseline
        for file in file_list:
            if os.path.exists(file):
                gpt2_helpful, gpt2_harmless = extract_avg_values(file)
                gpt2_helpful_vals.append(gpt2_helpful)
                gpt2_harmless_vals.append(gpt2_harmless)
            else:
                print(f"File not found: {file}")

        # Add jitter to the values
        gpt2_helpful_vals_jittered = add_jitter(np.array(gpt2_helpful_vals))
        gpt2_harmless_vals_jittered = add_jitter(np.array(gpt2_harmless_vals))

        # Calculate and print navigation efficiency
        nav_efficiency = calculate_navigation_efficiency(gpt2_helpful_vals, gpt2_harmless_vals, gpt2_helpful_ref, gpt2_harmless_ref)
        print(f"Navigation efficiency for {label}: {nav_efficiency:.2%}")

        # Plot the values for this baseline
        color = next(colors)
        marker = next(markers)
        scatter_plot = plt.scatter(gpt2_helpful_vals_jittered, gpt2_harmless_vals_jittered, color=color, marker=marker, s=15, edgecolor='none', alpha=0.6)
        handles.append(scatter_plot)

        # Calculate and plot the average point for each baseline
        avg_helpful = sum(gpt2_helpful_vals) / len(gpt2_helpful_vals)
        avg_harmless = sum(gpt2_harmless_vals) / len(gpt2_harmless_vals)
        if color == 'k':
            jitter_helpful = avg_helpful + np.random.normal(0, 0.02)
            jitter_harmless = avg_harmless + np.random.normal(0, 0.02)
        else:
            jitter_helpful = avg_helpful + np.random.normal(0, 0.0)
            jitter_harmless = avg_harmless + np.random.normal(0, 0.0)
        plt.scatter(jitter_helpful, jitter_harmless, color=color, marker=marker, s=120, alpha=0.9)

    # Plot a reference red circle
    plt.scatter(gpt2_helpful_ref, gpt2_harmless_ref, color='r', marker='o', s=120, label="Original model")

    x_min, x_max = -3.3, 0.5
    y_min, y_max = -2.0, 2.5

    # Define the band limits for horizontal and vertical reference lines
    y_band_lower = gpt2_harmless_ref - Ksigma
    y_band_upper = gpt2_harmless_ref
    x_band_lower = gpt2_helpful_ref - Ksigma
    x_band_upper = gpt2_helpful_ref
    plt.fill_between(x=[gpt2_helpful_ref, x_max], y1=y_band_lower, y2=y_band_upper, color='gray', alpha=0.2)
    plt.fill_betweenx(y=[gpt2_harmless_ref, y_max], x1=x_band_lower, x2=x_band_upper, color='gray', alpha=0.2)
    # plt.axhline(y=y_band_upper, color='gray', linestyle='--', linewidth=2)
    # plt.axvline(x=x_band_upper, color='gray', linestyle='--', linewidth=2)
    plt.hlines(y=y_band_upper, xmin=gpt2_helpful_ref, xmax=x_max, color='gray', linestyle='--', linewidth=2)
    plt.vlines(x=x_band_upper, ymin=gpt2_harmless_ref, ymax=y_max, color='gray', linestyle='--', linewidth=2)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15) 
    plt.xlabel("Helpfulness", fontsize=16)
    plt.ylabel("Harmlessness", fontsize=16)
    plt.title("Expected Reward (aka. Realized Value Level)", fontsize=16)

    # Custom legend
    legend_elements = [plt.Line2D([0], [0], marker=h.get_paths()[0], color='w', markerfacecolor=h.get_facecolor()[0], 
                                  markersize=10, alpha=1, label=label) 
                       for h, label in zip(handles, baselines.keys())]

    # Add the base model marker to the legend
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label="Original model"))

    plt.legend(handles=legend_elements, fontsize=12, loc='lower left')

    plt.savefig("results_comparison/fig_compare_avg_reward.pdf", bbox_inches='tight')
    print("Plot successfully saved to fig_compare_avg_reward.pdf!")


if __name__ == "__main__":

    # Directory paths
    base_model_file = 'results/opt1.3b-Anthropic-harmless.json'
    model_files = []

    # all DPO models
    # harmless_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # beta = 0.1
    # file_prefix = f"results_comparison/winrate_{beta}beta_DPO"
    # for ratio in harmless_ratios:
    #     model_files.append(f'modelsDPO/opt1.3b-2000sample-{beta}beta-{ratio}harmless-Anthropic-harmless.json')
    # win_rate_results_list = collect_multiple_results(model_files, base_model_file, file_prefix)
    # latex_table = render_latex_table(win_rate_results_list, file_prefix)
    # plot_helpful_vs_harmless(win_rate_results_list, harmless_ratios, file_prefix)


    # # all DPO soup models
    # harmless_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # beta = 0.5
    # file_prefix = f"results_comparison/winrate_{beta}beta_DPOsoup"
    # for ratio in harmless_ratios:
    #     model_files.append(f'modelsDPO/soup/opt1.3b-2000sample-{beta}beta-{ratio}soup-Anthropic-harmless.json')
    # win_rate_results_list = collect_multiple_results(model_files, base_model_file, file_prefix)
    # latex_table = render_latex_table(win_rate_results_list, file_prefix)
    # plot_helpful_vs_harmless(win_rate_results_list, harmless_ratios, file_prefix)


    # # all PPO models with Dirichlet random generated lambda
    # folder_paths = ['modelsPPO/random-lambda/', 'modelsPPO/random-lambda2/']
    # file_pattern = 'opt1.3b-Anthropic-harmless-lam_*_*-val_gpt2-helpful_gpt2-harmless-Anthropic-harmless.json'
    # model_files = []
    # for folder_path in folder_paths:
    #     full_pattern = os.path.join(folder_path, file_pattern)
    #     model_files.extend(glob.glob(full_pattern))  # Collect files from both folders
    # file_prefix = f"results_comparison/winrate_6scale_2valuesHH_PPO_DirichletRand"
    # win_rate_results_list = collect_multiple_results(model_files, base_model_file, file_prefix)


    # # all PPO models with MAP random generated feasible lambda
    # folder_paths = ['modelsPPO/MAP-lambda/', 'modelsPPO/MAP-lambda2/']
    # file_pattern = 'opt1.3b-Anthropic-harmless-lam_*_*-val_gpt2-helpful_gpt2-harmless-Anthropic-harmless.json'
    # model_files = []
    # for folder_path in folder_paths:
    #     full_pattern = os.path.join(folder_path, file_pattern)
    #     model_files.extend(glob.glob(full_pattern))  # Collect files from both folders
    # file_prefix = f"results_comparison/winrate_6scale_2valuesHH_PPO_MapRand"
    # win_rate_results_list = collect_multiple_results(model_files, base_model_file, file_prefix)


    # # all decoding results with Dirichlet random generated lambda
    # folder_paths = ['modelsDecoding/random-lambda/']
    # file_pattern = 'opt1.3b-Anthropic-harmless_lam=*,*_val=gpt2-helpful,gpt2-harmless.json'
    # model_files = []
    # for folder_path in folder_paths:
    #     full_pattern = os.path.join(folder_path, file_pattern)
    #     model_files.extend(glob.glob(full_pattern))  # Collect files from both folders
    # file_prefix = f"results_comparison/winrate_6scale_2valuesHH_Decoding_DirichletRand"
    # win_rate_results_list = collect_multiple_results(model_files, base_model_file, file_prefix, metrics=['coherence', 'diversity', 'gpt2-harmless', 'gpt2-helpful', 'humor'])


    # # all decoding results with MAP random generated feasible lambda
    # folder_paths = ['modelsDecoding/MAP-lambda/']
    # file_pattern = 'opt1.3b-Anthropic-harmless_lam=*,*_val=gpt2-helpful,gpt2-harmless.json'
    # model_files = []
    # for folder_path in folder_paths:
    #     full_pattern = os.path.join(folder_path, file_pattern)
    #     model_files.extend(glob.glob(full_pattern))  # Collect files from both folders
    # file_prefix = f"results_comparison/winrate_6scale_2valuesHH_Decoding_MapRand"
    # win_rate_results_list = collect_multiple_results(model_files, base_model_file, file_prefix, metrics=['coherence', 'diversity', 'gpt2-harmless', 'gpt2-helpful', 'humor'])

    plot_winrate()
    plot_cLevels()
