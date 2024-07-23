import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from utils import ALL_SUPPORTED_VALUES, ALL_SUPPORTED_VALUES_plotnames
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.io as pio   
pio.kaleido.scope.mathjax = None
import sys

def quantile_transform(value1, value2, filename, x_list, y_list):
    # Load the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Extract values for the specified keys
    value1_data = [entry[value1] for entry in data]
    value2_data = [entry[value2] for entry in data]
    
    # Sort the data once for quantile calculations
    sorted_value1_data = np.sort(value1_data)
    sorted_value2_data = np.sort(value2_data)

    # Calculate quantiles for x_list and y_list
    x_quantiles = [np.searchsorted(sorted_value1_data, x) / len(value1_data) for x in x_list]
    y_quantiles = [np.searchsorted(sorted_value2_data, y) / len(value2_data) for y in y_list]
    return x_quantiles, y_quantiles

def plot_pareto(column_names, csv_path, alignment_data=None, reward_filepath=None, use_quantile_transform=True):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Precompute the quantiles for the entire DataFrame if needed
    if use_quantile_transform:
        x_list, y_list = df[column_names[0]].tolist(), df[column_names[1]].tolist()
        x_quantiles, y_quantiles = quantile_transform(column_names[0], column_names[1], reward_filepath, x_list, y_list)
        df[column_names[0]] = x_quantiles
        df[column_names[1]] = y_quantiles
        
    # Extract the two columns to plot
    x_values = df[column_names[0]]
    y_values = df[column_names[1]]

    # Create a square-shaped scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x_values, y_values, color='blue', alpha=0.5, s=12, edgecolors='none')
    
    point_colors = ['red', 'green', 'purple', 'orange']
    point_markers = ['x', 'o', '^', 's']  # Cross, Circle, Triangle-up, Square

    def point_trans(points):
        if use_quantile_transform:
            qx, qy = quantile_transform(column_names[0], column_names[1], reward_filepath, [points[0]], [points[1]])
            qx, qy = qx[0], qy[0]
        else:
            qx, qy = points
        return qx, qy

    if alignment_data:
        # Plot alignment points and draw arrows
        for i, (name, points) in enumerate(alignment_data.items()):
            qx, qy = point_trans(points)
            plt.scatter(qx, qy, color=point_colors[i % len(point_colors)], marker=point_markers[i % len(point_markers)], s=50, label=name)
        
        # Draw arrows from pre-alignment to other alignments if provided
        pre_alignment = alignment_data["Pre-alignment"]
        qx_pre, qy_pre = point_trans(pre_alignment)

        arrow_properties = {
            "Sole-Helpful-alignment": {"color": "black", "style": ":"},
            "Sole-Diversity-alignment": {"color": "black", "style": "-"},
            "MAP-alignment": {"color": "black", "style": "-."}
        }
        for name, points in alignment_data.items():
            if points and name != "Pre-alignment":
                qx, qy = point_trans(points)

                # Calculate the scale factor for arrow size based on quantile difference
                dx, dy = qx - qx_pre, qy - qy_pre

               # Calculate the arrow scale factor and length dynamically
                scale = min(0.005, max(abs(dx), abs(dy)))
                head_width = max(0.01, scale * 0.5)
                head_length = max(0.015, scale * 1)
                
                # Use FancyArrowPatch for better control and appearance
                arrow = FancyArrowPatch((qx_pre, qy_pre), (qx_pre + dx, qy_pre + dy),
                                         color=arrow_properties[name]["color"], arrowstyle='-|>',
                                         mutation_scale=20, linestyle=arrow_properties[name]["style"])
                plt.gca().add_patch(arrow)

    # Add legend for alignments
    # Create a proxy artist for the blue dot and add it to the legend
    blue_dot = Line2D([0], [0], marker='o', color='w', label='Feasible alignments', markerfacecolor='blue', markersize=6)
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [blue_dot], loc='lower left')

    x_label = ALL_SUPPORTED_VALUES_plotnames[ALL_SUPPORTED_VALUES.index(column_names[0])]
    y_label = ALL_SUPPORTED_VALUES_plotnames[ALL_SUPPORTED_VALUES.index(column_names[1])]

    # Title and labels
    plt.title("Value Palette", fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xlim([0.45, 1.04])
    plt.ylim([0, 1.04])
    plt.grid(False)
    
    # Save the plot to a PDF file under results/
    pdf_path = f"results/pareto_temp_{','.join(column_names)}.pdf"
    plt.savefig(pdf_path, format='pdf')
    print(f"Plot saved to {pdf_path}")

    # Show the plot
    plt.show()


def plot_matrix_scatterplot(csv_path):
    '''
        Intend to input a path to randomly generated c values to show Tradeoffs and Pareto Fronts
    '''
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Use Seaborn's pairplot to create a 7x6 matrix of scatter plots
    sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.5})
    
    # Save the plot to a PDF file under results/
    pdf_path = csv_path.replace('.csv', '_matrix_scatterplot.pdf')
    plt.savefig(pdf_path, format='pdf')
    print(f"Plot saved to {pdf_path}")

    # Show the plot
    plt.show()


def plot_hist(json_file):
    '''
        Intend to input a path to sentences along with calculated rewards to show the distribution of each reward
    '''

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


def plot_hist_positive(json_file):
    '''
        Intends to input a path to a JSON file with a "positive" column and plot the distribution of these values.
    '''
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


def plot_radar_chart(categories, model_scores, title_text, output_file):
    # Convert the dictionary into a DataFrame
    data = []
    for model, scores in model_scores.items():
        for category, score in zip(categories, scores):
            data.append({"Palette": model, "category": category, "score": score})
    df_scores = pd.DataFrame(data)
    
    # Create the radar chart
    fig = px.line_polar(df_scores, r='score', theta='category', line_close=True,
                        color='Palette', markers=True,
                        category_orders={'category': categories})
    

    # Update the layout for larger fonts
    fig.update_layout(
        font=dict(size=18),  # Sets the global font size
        legend=dict(font=dict(size=16, color="black"), orientation='h', x=0.5, y=-0.1, xanchor='center', yanchor='top'),
        polar=dict(  # Configuring polar layout
            radialaxis=dict(showticklabels=True, tickfont=dict(size=16)),
            angularaxis=dict(tickfont=dict(size=16))
        ),
        margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins if necessary
        title=dict(text=title_text, x=0.5, y=0.95, xanchor='center', yanchor='top', font=dict(size=20))  # Adjust title position
    )
    
    # Increase the figure size if needed
    fig.update_layout(width=800, height=600)
    
    # Save to PDF
    fig.write_image(output_file, engine='kaleido')
    return
    # return fig.show()


if __name__ == '__main__':


    '''
        Example usage of hist inspection:
    '''
    # json_file = "results/Llama27b-chat-Anthropic-harmless.json"
    # json_file = "results/opt1.3b-Anthropic-harmless.json"
    json_file = "results/opt1.3b-Imdb.json"
    # plot_hist(json_file)
    # plot_hist_positive(json_file)

    csv_path = "results/opt1.3b-Imdb_pareto-1_randLam_val=all_numericC-levels.csv"
    plot_matrix_scatterplot(csv_path)
    sys.exit("debug")

    '''
        Example usage of show pareto front:
    '''

    # plot_pareto_all(csv_path)

    # csv_path = "results/pareto_-1_align_gpt2-helpful,gpt2-harmless_assess_gpt2-helpful,gpt2-harmless.csv"
    # column_names = ["gpt2-helpful", "gpt2-harmless"]
    # alignment_data = {
    #     "Pre-alignment": [-1.011,1.248],
    #     "MAP-alignment": [-0.462,1.885],
    #     "Sole-Helpful-alignment": [-0.462,0.981],
    #     "Sole-Harmless-alignment": [-1.32,1.885],
    # }

    # csv_path = "results/pareto_-1_align_gpt2-helpful,diversity_assess_gpt2-helpful,diversity.csv"
    # column_names = ["gpt2-helpful", "diversity"]
    # alignment_data = {
    #     "Pre-alignment": [-1.011,0.848],
    #     "MAP-alignment": [-0.462,0.919],
    #     "Sole-Helpful-alignment": [-0.462,0.846],
    #     "Sole-Diversity-alignment": [-1.068,0.919],
    # }
    # plot_pareto(column_names, csv_path, alignment_data, reward_filepath="results/Llama27b-chat-Anthropic-harmless.json", use_quantile_transform=True)


    '''
        Results for basemodel_name: "Llama27b-chat", data_name: "Anthropic-harmless"
    '''
    # renaming ALL_SUPPORTED_VALUES = ["humor", "gpt2-helpful", "gpt2-harmless", "diversity", "coherence", "perplexity"]
    # level of preference
    categories = ["Humor", "Helpful", "Harmless", "Diversity", "Coherence", "Perplexity"]
    rewards_llama2chat_conversation_MC = {
        "Pre-align": [0.422,0.526,0.454,0.328,0.464,0.426],
        "MAP-align to 50% quantile": [0.470,0.507,0.498,0.316,0.475,0.433],
        "MAP-align to 60% quantile": [0.557,0.551,0.539,0.292,0.494,0.432],
        "MAP-align to 70% quantile": [0.651,0.573,0.574,0.276,0.522,0.426],
        "MAP-align to 80% quantile": [0.702,0.549,0.611,0.269,0.531,0.429],
    }
    # plot_radar_chart(categories, model_scores=rewards_llama2chat_conversation_MC, title_text='Using Model-Generated Texts', output_file="results/rewards_llama2chat_conversation_MC.pdf")

    rewards_llama2chat_conversation_numeric = {
        "Pre-align": [0.422,0.526,0.454,0.328,0.464,0.426],
        "MAP-align to 50% quantile": [0.500,0.526,0.500,0.328,0.475,0.427],
        "MAP-align to 60% quantile": [0.600,0.600,0.600,0.344,0.496,0.429],
        "MAP-align to 70% quantile": [0.700,0.700,0.700,0.379,0.513,0.429],
        "MAP-align to 80% quantile": [0.801,0.800,0.800,0.477,0.538,0.425],
    }
    plot_radar_chart(categories, model_scores=rewards_llama2chat_conversation_numeric, title_text='Using Numeric Estimates', output_file="results/rewards_llama2chat_conversation_numeric.pdf")
