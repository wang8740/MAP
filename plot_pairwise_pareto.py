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


def plot_pareto(column_names, csv_path, alignment_data=None, reward_filepath=None, use_quantile_transform=True, xlim=[0,1.04], ylim=[0,1.04]):
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
    point_markers = ['*', 'o', 'x', 's']  # Cross, Circle, Triangle-up, Square

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
            plt.scatter(qx, qy, color=point_colors[i % len(point_colors)], marker=point_markers[i % len(point_markers)], s=70, label=name)
        
        # Draw arrows from pre-alignment to other alignments if provided
        pre_alignment = alignment_data["Original model"] # alignment_data["Pre-alignment"]
        qx_pre, qy_pre = point_trans(pre_alignment)

        arrow_properties = {
            "Sole-Helpfulness": {"color": "orange", "style": ":"},
            "Sole-Harmlessness": {"color": "purple", "style": "-"},
            "Sole-Humor": {"color": "purple", "style": "--"},
            "MAP": {"color": "green", "style": "-."}
        }
        for name, points in alignment_data.items():
            if points and name != "Original model": # "Pre-alignment":
                qx, qy = point_trans(points)

                # Calculate the scale factor for arrow size based on quantile difference
                dx, dy = qx - qx_pre, qy - qy_pre

               # Calculate the arrow scale factor and length dynamically
                scale = min(0.005, max(abs(dx), abs(dy)))
                head_width = max(0.01, scale * 0.3)
                head_length = max(0.015, scale * 1)
                
                # Use FancyArrowPatch for better control and appearance
                arrow = FancyArrowPatch((qx_pre, qy_pre), (qx_pre + dx, qy_pre + dy),
                                         color=arrow_properties[name]["color"], arrowstyle='-|>',
                                         mutation_scale=20, linestyle=arrow_properties[name]["style"])
                plt.gca().add_patch(arrow)

    # Add legend for alignments
    # Create a proxy artist for the blue dot and add it to the legend
    blue_dot = Line2D([0], [0], marker='o', color='w', label='Randomly aligned model', markerfacecolor='blue', markersize=6)
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [blue_dot], loc='lower left')

    x_label = ALL_SUPPORTED_VALUES_plotnames[ALL_SUPPORTED_VALUES.index(column_names[0])]
    y_label = ALL_SUPPORTED_VALUES_plotnames[ALL_SUPPORTED_VALUES.index(column_names[1])]

    # Title and labels
    plt.title("Expected Reward (aka. Realized Value Level)", fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(False)
    
    # Save the plot to a PDF file under results/
    pdf_path = f"results/fig_pareto_{','.join(column_names)}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {pdf_path}")

    # Show the plot
    plt.show()


if __name__ == '__main__':


    # json_file = "results/llama2_chat-Anthropic-harmless.json"
    # json_file = "results/opt1.3b-Anthropic-harmless.json"
    # json_file = "results/opt1.3b-Imdb.json"

    '''
        Inspect correlation among all values in terms of c-levels calculated from random lambda
    '''
    # plot_hist(json_file)
    # plot_hist_positive(json_file)

    # csv_path = "results/opt1.3b-Imdb_pareto-1_randLam_val=all_numericC-levels.csv"
    # plot_matrix_scatterplot(csv_path)
    # sys.exit("debug")

    '''
        Inspect correlation among pair of values in terms of sentence-level rewards
        NOTE: results are not informative, so ignore it for now
    '''
    # # Load the JSON file into a DataFrame
    # json_file = "results/llama2_chat-Anthropic-harmless.json"
    # base_name = os.path.basename(json_file)  # Get the filename from the path
    # core_name = base_name.replace('.json', '')  # Remove the .json extension

    # data = pd.read_json(json_file)

    # # Only keeping relevant columns
    # columns_of_interest = ['gpt2-helpful', 'humor', 'gpt2-harmless']
    # df = data[columns_of_interest]

    # # Use Seaborn's pairplot to create scatter plots between "gpt2-helpful, humor" and "gpt2-helpful, gpt2-harmless"
    # sns.pairplot(df, vars=['gpt2-helpful', 'humor', 'gpt2-harmless'], diag_kind='kde', plot_kws={'alpha': 0.5})

    # # Save the plot to a PDF file under results/
    # pdf_path = os.path.join('results', f'fig_{core_name}_scatterplot.pdf')
    # plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    # plt.close()
    # print(f"Plot saved to {pdf_path}")
    # sys.exit("debug")

 
    '''
        Pareto Step 1: Generate csv file that contains three values to align, using a specific model
    '''
    model = "llama2_chat" # llama2_chat, opt1.3b
    scaling = -1 #-1, 1, 3
    a, b, c = "humor", "gpt2-helpful", "gpt2-harmless" # "diversity"
    # from rewardProcessor import RewardProcessor
    # r = RewardProcessor(values_to_evaluate=f"{a},{b},{c}", values_to_align=f"{a},{b},{c}", file_path=f"results/{model}-Anthropic-harmless.json")
    # r.assess_postalignment_multivalue(k=500, scaling=scaling, scaling_MAX=4)

    '''
        Pareto Step 2: get alignment_data
    '''
    from alignValues import AlignValues

    ## original c-levels: 
    # humor,gpt2-helpful,gpt2-harmless
    # avg,0.604,-1.011,1.248
    # 80%,2.363,-0.108,2.164
    # optimized_lambda, success = AlignValues(value_list="humor,gpt2-helpful,gpt2-harmless", c_list=[2.363,-0.108,2.164], file_path=f"results/{model}-Anthropic-harmless.json").optimize_lambda()
    # print(f"Optimized lambda for 80% joint: {optimized_lambda}, success: {success}") #[5.925, 2.443, 2.923]
    # optimized_lambda, success = AlignValues(value_list="humor", c_list=[2.363], file_path=f"results/{model}-Anthropic-harmless.json").optimize_lambda()
    # print(f"Optimized lambda for 80% humor: {optimized_lambda}, success: {success}") #[2.887]
    # optimized_lambda, success = AlignValues(value_list="gpt2-helpful", c_list=[-0.108], file_path=f"results/{model}-Anthropic-harmless.json").optimize_lambda()
    # print(f"Optimized lambda for 80% gpt2-helpful: {optimized_lambda}, success: {success}") #[0.693]
    # optimized_lambda, success = AlignValues(value_list="gpt2-harmless", c_list=[2.164], file_path=f"results/{model}-Anthropic-harmless.json").optimize_lambda()
    # print(f"Optimized lambda for 80% gpt2-harmless: {optimized_lambda}, success: {success}") #[0.988]

    # RewardProcessor(values_to_evaluate="humor,gpt2-helpful,gpt2-harmless", values_to_align="humor,gpt2-helpful,gpt2-harmless", file_path=f"results/{model}-Anthropic-harmless.json").assess_postalignment_multivalue(lam=[5.925, 2.443, 2.923])
    # 2.363,-0.108,2.164
    # RewardProcessor(values_to_evaluate="humor,gpt2-helpful,gpt2-harmless", values_to_align="humor", file_path=f"results/{model}-Anthropic-harmless.json").assess_postalignment_multivalue(lam=[2.887])
    # 2.363,-1.312,0.938
    # RewardProcessor(values_to_evaluate="humor,gpt2-helpful,gpt2-harmless", values_to_align="gpt2-helpful", file_path=f"results/{model}-Anthropic-harmless.json").assess_postalignment_multivalue(lam=[0.693])
    # 0.2,-0.108,0.795
    # RewardProcessor(values_to_evaluate="humor,gpt2-helpful,gpt2-harmless", values_to_align="gpt2-harmless", file_path=f"results/{model}-Anthropic-harmless.json").assess_postalignment_multivalue(lam=[0.988])
    # 0.603,-1.462,2.164
    
    '''
        Pareto Step 3: Plot the pareto tradeoffs
    '''
    csv_path = f"results/{model}-Anthropic-harmless_pareto{scaling}_randLam_val={a},{b},{c}_numericC-levels.csv"

    alignment_harmless_helpful = {
        "Original model": [1.248,-1.011],
        "MAP": [2.164,-0.108],
        "Sole-Harmlessness": [2.164,-1.462],
        "Sole-Helpfulness": [0.795,-0.108],
    }
    plot_pareto(column_names=[c, b], csv_path=csv_path, alignment_data=alignment_harmless_helpful, reward_filepath=f"results/{model}-Anthropic-harmless.json") #, xlim=[0.2,0.8], ylim=[0.3,0.9])

    alignment_humor_helpful = {
        "Original model": [0.604,-1.011],
        "MAP": [2.363,-0.108],
        "Sole-Humor": [2.363,-1.312],
        "Sole-Helpfulness": [0.2,-0.108],
    }
    plot_pareto(column_names=[a, b], csv_path=csv_path, alignment_data=alignment_humor_helpful, reward_filepath=f"results/{model}-Anthropic-harmless.json") #, xlim=[0.3,0.7], ylim=[0.3,0.9])
    # plot_pareto(column_names=[b, c], csv_path=csv_path, alignment_data=None, reward_filepath=f"results/{model}-Anthropic-harmless.json", use_quantile_transform=True)

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
