# Goal of this script is to visualize the subregion of MAP-feasible lambda and how it can shrink as more values are considered

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from matplotlib import rc

# Enable LaTeX rendering
rc('text', usetex=True)

# Ensure that LaTeX can handle bold math symbols
rc('text.latex', preamble=r'\usepackage{bm}')

# cannot use interactive mode because of headless server
# plt.switch_backend('TkAgg')

def plot_lambdas_3D(
    file_path: str, 
    x_value: str, 
    y_value: str, 
    z_value: str, 
    output_prefix: str
) -> None:
    """Plot a 3D scatter plot comparing feasible and random lambda values for three specified metrics.

    This function generates a 3D scatter plot using optimized and Dirichlet (random) lambda values for 
    three specified metrics. The plot helps visualize the subregion of MAP-feasible lambdas for models 
    aligned to specific human values.

    Args:
        file_path (str): Path to the CSV file containing lambda values for the specified metrics.
        x_value (str): Metric name for the x-axis (e.g., 'gpt2-helpful').
        y_value (str): Metric name for the y-axis (e.g., 'gpt2-harmless').
        z_value (str): Metric name for the z-axis (e.g., 'humor').
        output_prefix (str): Prefix for the output PDF file where the plot will be saved.

    Example:
        >>> plot_lambdas_3D("plot_rand_lambda_6scale_3D.csv", "gpt2-helpful", "gpt2-harmless", "humor", "results/lambda_3D_plot")
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Retrieve the index of the x_value, y_value, z_value from the "values" column
    values_list = df['values'].iloc[0].split(',')
    try:
        x_idx = values_list.index(x_value)
        y_idx = values_list.index(y_value)
        z_idx = values_list.index(z_value)
    except ValueError as e:
        raise ValueError(f"Error: One of the values '{x_value}', '{y_value}', or '{z_value}' not found in file '{file_path}'.") from e

    # Extract the optimized_lambda and Dirichlet_lambda_ref columns
    lambda_values = df['optimized_lambda'].apply(lambda x: list(map(float, x.split(','))))
    dirichlet_lambda_values = df['Dirichlet_lambda_ref'].apply(lambda x: list(map(float, x.split(','))))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates for optimized_lambda based on the retrieved indices
    xs_optimized = [l[x_idx] for l in lambda_values]
    ys_optimized = [l[y_idx] for l in lambda_values]
    zs_optimized = [l[z_idx] for l in lambda_values]

    # Extract x, y, z coordinates for Dirichlet_lambda_ref based on the retrieved indices
    xs_dirichlet = [l[x_idx] for l in dirichlet_lambda_values]
    ys_dirichlet = [l[y_idx] for l in dirichlet_lambda_values]
    zs_dirichlet = [l[z_idx] for l in dirichlet_lambda_values]

    # Plot optimized_lambda points in blue with semi-transparency
    ax.scatter(xs_optimized, ys_optimized, zs_optimized, c='b', marker='o', alpha=0.6, s=8, label=fr'Feasible $\lambda$')

    # Plot Dirichlet_lambda_ref points in red with semi-transparency
    ax.scatter(xs_dirichlet, ys_dirichlet, zs_dirichlet, c='k', marker='x', alpha=0.5, s=8, label=fr'Random $\lambda$')

    ax.set_title(f'(c) Align Helpfulness, Harmlessness, and Humor', fontsize=16)

    # Set labels with LaTeX formatting and fontsize 16
    ax.set_xlabel(fr'$\lambda_{{\mathrm{{Helpfulness}}}}$', fontsize=18)
    ax.set_ylabel(fr'$\lambda_{{\mathrm{{Harmlessness}}}}$', fontsize=18)
    ax.set_zlabel(fr'$\lambda_{{\mathrm{{Humor}}}}$', fontsize=18)

    # Set axis ranges for x, y, z to be 0-6
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_zlim(0, 6)

    # Ensure ticks are integers only
    ax.set_xticks(range(0, 7))
    ax.set_yticks(range(0, 7))
    ax.set_zticks(range(0, 7))

    # Add legend
    ax.legend(prop={'size': 16})

    # Save the plot to a PDF
    plt.savefig(f"{output_prefix}.pdf", bbox_inches='tight')
    print(f"Successfully saved plot to {output_prefix}.pdf!")

    plt.show()


def plot_lambdas_2D_subplots(
    file_paths: list[str], 
    x_value: str, 
    y_value: str, 
    output_prefix: str
) -> None:
    """Generate 2D subplots for lambda values, with one subplot for each CSV file.

    This function creates multiple 2D scatter plots for optimized and Dirichlet (random) lambda values 
    for two specified metrics, displaying one subplot per CSV file. This helps to analyze the shrinkage 
    of feasible lambda regions as more values are considered.

    Args:
        file_paths (list[str]): List of CSV file paths containing lambda values for each subplot.
        x_value (str): Metric name for the x-axis (e.g., 'gpt2-helpful').
        y_value (str): Metric name for the y-axis (e.g., 'gpt2-harmless').
        output_prefix (str): Prefix for the output PDF file where the plot will be saved.

    Example:
        >>> file_paths = ["lambda_data_2D.csv", "lambda_data_3D.csv"]
        >>> plot_lambdas_2D_subplots(file_paths, "gpt2-helpful", "gpt2-harmless", "results/lambda_2D_subplots")
    """

    # Create a figure with subplots (one for each file, horizontal layout)
    num_files = len(file_paths)
    fig, axes = plt.subplots(1, num_files, figsize=(5*num_files, 5))

    # Define a color cycle to use different colors for each file
    colors = itertools.cycle(['orange', 'b', 'r', 'c', 'm', 'y'])
    D = [2 + d for d in range(len(file_paths))]

    # Loop over the files and create a subplot for each file
    for idx, file_path in enumerate(file_paths):
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Retrieve the index of the x_value and y_value from the "values" column
        values_list = df['values'].iloc[0].split(',')
        x_idx = values_list.index(x_value)
        y_idx = values_list.index(y_value)

        # Extract the optimized_lambda column
        lambda_values = df['optimized_lambda'].apply(lambda x: list(map(float, x.split(','))))

        # Extract x, y coordinates for optimized_lambda based on the retrieved indices
        xs_optimized = [l[x_idx] for l in lambda_values]
        ys_optimized = [l[y_idx] for l in lambda_values]

        # Get the Dirichlet_lambda_ref for this file
        dirichlet_lambda_values = df['Dirichlet_lambda_ref'].apply(lambda x: list(map(float, x.split(','))))
        xs_dirichlet = [l[x_idx] for l in dirichlet_lambda_values]
        ys_dirichlet = [l[y_idx] for l in dirichlet_lambda_values]

        # Get the next color from the cycle
        color = next(colors)

        # Plot on the corresponding subplot (axes[idx])
        ax = axes[idx] if num_files > 1 else axes

        # Plot optimized_lambda points with semi-transparency
        ax.scatter(xs_optimized, ys_optimized, c=color, marker='o', alpha=1, s=15,
                   label=r'Desirable $\boldsymbol{\lambda}$') #(align {D[idx]} values)

        # Plot Dirichlet_lambda_ref points specific to this file
        ax.scatter(xs_dirichlet, ys_dirichlet, c='b', marker='x', alpha=0.7, s=12,
                   label=r'Random $\boldsymbol{\lambda}$')

        # Use shorter or wrapped title if too long
        title_text = '(b) Align Helpfulness, Harmlessness, and Humor'
        ax.set_title(title_text, fontsize=16, wrap=True)

        # Set labels with LaTeX formatting and fontsize 16
        ax.set_xlabel(fr'$\lambda_{{\mathrm{{Helpfulness}}}}$', fontsize=16)
        ax.set_ylabel(fr'$\lambda_{{\mathrm{{Harmlessness}}}}$', fontsize=16)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.legend(prop={'size': 16})

    # Add margins to avoid truncation
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)

    # Adjust layout and save the plot to a PDF
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.pdf", bbox_inches='tight')
    print(f"Successfully saved plot to {output_prefix}.pdf!")


if __name__ == "__main__":
    # 3D
    # file_path = 'plot_rand_lambda_6scale_3D.csv'
    # x_value = 'gpt2-helpful'
    # y_value = 'gpt2-harmless'
    # z_value = 'humor'
    # plot_lambdas_3D(file_path, x_value, y_value, z_value, output_prefix='results/plot_rand_lambda_MAP_region_(c)')

    # # 2D
    x_value = 'gpt2-helpful'
    y_value = 'gpt2-harmless'
    # file_paths = [f'plot_rand_lambda_6scale_{D}D.csv' for D in range(2,7)]

    file_paths = ['plot_rand_lambda_6scale_2D.csv'] # (a)
    output_prefix = 'results/plot_rand_lambda_MAP_region_(a)'

    # file_paths = ['plot_rand_lambda_6scale_3D.csv'] # (b)
    # output_prefix = 'results/plot_rand_lambda_MAP_region_(b)'

    plot_lambdas_2D_subplots(file_paths, x_value, y_value, output_prefix)

