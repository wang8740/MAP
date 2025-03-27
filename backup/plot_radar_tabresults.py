import pandas as pd
from utils import ALL_SUPPORTED_VALUES, ALL_SUPPORTED_VALUES_plotnames
import plotly.express as px
import plotly.io as pio   
pio.kaleido.scope.mathjax = None
from rewardProcessor import RewardProcessor
import itertools


def plot_radar_chart(
    categories: list[str], 
    model_scores: dict[str, list[float]], 
    title_text: str, 
    output_file: str, 
    model_name: str, 
    art_set: int = 0
) -> None:
    """Generate and save a radar chart to visualize model alignment scores across multiple categories.

    This function creates a radar chart to display alignment scores for various model configurations
    across specified categories. Each model's scores are transformed using a quantile transformation for
    comparison, and custom symbols and colors are applied to distinguish between models. 

    Args:
        categories (list[str]): List of categories to display on the radar chart (e.g., 'Humor', 'Helpfulness').
        model_scores (dict[str, list[float]]): Dictionary mapping model names to their scores in each category.
        title_text (str): Title for the radar chart.
        output_file (str): Path for saving the radar chart as a PDF file.
        model_name (str): Name of the model to fetch the appropriate file for score transformation.
        art_set (int, optional): Specifies the visual style (markers and colors) of the radar chart. Defaults to 0.

    Example:
        >>> categories = ["Humor", "Helpfulness", "Harmlessness", "Diversity", "Coherence", "Perplexity"]
        >>> model_scores = {
        ...     "Original model": [2.07, -1.471, 0.245, 0.876, 0.434, -3.337],
        ...     "MAP-align to Humor-80%": [2.516, -1.419, 0.012, 0.889, 0.429, -3.205],
        ...     "MAP-align to Helpfulness-80%": [1.992, -0.754, -0.350, 0.880, 0.427, -3.196]
        ... }
        >>> plot_radar_chart(
        ...     categories, model_scores, 
        ...     title_text='Model Alignment Comparison', 
        ...     output_file="results/fig_radar_example.pdf", 
        ...     model_name="opt1.3b", 
        ...     art_set=1
        ... )
    """

    file_path = f"results/{model_name}-Anthropic-harmless.json"
    reward_processor = RewardProcessor(file_path=file_path)
    
    # Convert the dictionary into a DataFrame with quantile transformed scores
    data = []
    for model, scores in model_scores.items():
        transformed_scores = reward_processor.quantile_transform_single_c(scores)
        for category, score in zip(categories, transformed_scores):
            data.append({"Palette": model, "category": category, "score": score})
    df_scores = pd.DataFrame(data)
    
    # Create the radar chart
    fig = px.line_polar(df_scores, r='score', theta='category', line_close=True,
                        color='Palette', markers=True,
                        category_orders={'category': categories})

    # Define marker symbols and colors for each model
    if art_set==0:
        marker_symbols = itertools.cycle(['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star', 'pentagon', 'hexagon'])
        color_set = itertools.cycle(['blue', 'orange', 'red', 'black', 'green', 'cyan', 'navy', 'darkorange', 'purple', 'brown'])
    elif art_set==1:
        marker_symbols = itertools.cycle(['circle', 'square', 'diamond', 'cross', 'x'])
        color_set = itertools.cycle(['blue', 'orange', 'red', 'black', 'green'])
    if art_set==2:
        marker_symbols = itertools.cycle(['circle', 'star', 'pentagon', 'hexagon'])
        color_set = itertools.cycle(['blue', 'navy', 'darkorange', 'purple', 'brown'])
    else:
        pass

    # Update each trace with a different marker symbol and color
    for trace, marker, color in zip(fig.data, marker_symbols, color_set):
        trace.update(marker=dict(symbol=marker, size=12), line=dict(color=color, width=2))
    
    fig.update_layout(
        font=dict(size=18),
        legend=dict(
            font=dict(size=16, color="black"),
            orientation='h',
            x=0.5, 
            y=-0.2,  # Adjust the position to fit under the chart
            xanchor='center', 
            yanchor='top',
            tracegroupgap=10,  # Adjust vertical space between legend items
            itemwidth=80  # Adjust the width of each item to fit more on one row
        ),
        polar=dict(
            radialaxis=dict(showticklabels=True, tickfont=dict(size=16), range=[0, 0.8]),
            angularaxis=dict(tickfont=dict(size=16))
        ),
        margin=dict(l=40, r=40, t=90, b=40),
        title=dict(text=title_text, x=0.5, y=0.95, xanchor='center', yanchor='top', font=dict(size=20))
    )
    fig.update_layout(width=800, height=600)
    fig.write_image(output_file, engine='kaleido')
    
    return


if __name__ == '__main__':

    '''
        Results for basemodel_name: "opt1.3b", "llama2_chat", data_name: "Anthropic-harmless"
    '''

    categories = ["Humor", "Helpful", "Harmless", "Diversity", "Coherence", "Perplexity"]

    # Opt-1.3 combined
    # rewards_opt1_3b_conversation_MC = {
    #     "Pre-align": [2.07, -1.471, 0.245, 0.876, 0.434, -3.337],
    #     "MAP-align to Humor-80%": [2.516, -1.419, 0.012, 0.889, 0.429, -3.205],
    #     "MAP-align to Helpful-80%": [1.992, -0.754, -0.350, 0.880, 0.427, -3.196],
    #     "MAP-align to Harmless-80%": [1.970, -1.864, 0.968, 0.877, 0.417, -3.166],
    #     "MAP-align to HHH-50%": [2.437, -1.382, 0.207, 0.884, 0.426, -3.170],
    #     "MAP-align to HHH-60%": [2.476, -1.325, 0.481, 0.883, 0.433, -3.149],
    #     "MAP-align to HHH-70%": [2.494, -1.287, 0.661, 0.879, 0.446, -3.136],
    # }
    # plot_radar_chart(categories, model_scores=rewards_opt1_3b_conversation_MC, title_text='Value Palletes for Opt-1.3B Model', output_file="results/fig_radar_opt1_3b_conversation_MC.pdf", model_name="opt1.3b")

    # Opt-1.3 Separated into (a) (b)
    rewards_opt1_3b_conversation_MC_a = {
        "Original model": [2.07, -1.471, 0.245, 0.876, 0.434, -3.337],
        "MAP to HHH-50%": [2.437, -1.382, 0.207, 0.884, 0.426, -3.170],
        "MAP to HHH-60%": [2.476, -1.325, 0.481, 0.883, 0.433, -3.149],
        "MAP to HHH-70%": [2.494, -1.287, 0.661, 0.879, 0.446, -3.136],
    }
    rewards_opt1_3b_conversation_MC_b = {
        "Original model": [2.07, -1.471, 0.245, 0.876, 0.434, -3.337],
        "MAP to Humor-80%": [2.516, -1.419, 0.012, 0.889, 0.429, -3.205],
        "MAP to Helpfulness-80%": [1.992, -0.754, -0.350, 0.880, 0.427, -3.196],
        "MAP to Harmlessness-80%": [1.970, -1.864, 0.968, 0.877, 0.417, -3.166],
    }
    plot_radar_chart(categories, model_scores=rewards_opt1_3b_conversation_MC_a, title_text='(a) Multi-value Palette Alignment', output_file="results/fig_radar_opt1_3b_conversation_MC_a.pdf", model_name="opt1.3b", art_set=1)
    plot_radar_chart(categories, model_scores=rewards_opt1_3b_conversation_MC_b, title_text='(b) Single-value Palette Alignment', output_file="results/fig_radar_opt1_3b_conversation_MC_b.pdf", model_name="opt1.3b", art_set=2)

    # llama2chat combined
    # rewards_llama2chat_conversation_MC = {
    #     "Pre-align": [0.604, -1.011, 1.248, 0.848, 0.521, -1.375],
    #     "MAP-align to Humor-80%": [2.208, -1.257, 1.055, 0.815, 0.552, -1.469],
    #     "MAP-align to Helpful-80%": [0.501, -0.391, 0.941, 0.856, 0.527, -1.362],
    #     "MAP-align to Harmless-80%": [0.470, -1.328, 1.911, 0.859, 0.501, -1.335],
    #     "MAP-align to HHH-50%": [0.984, -1.057, 1.362, 0.843, 0.527, -1.367],
    #     "MAP-align to HHH-60%": [1.564, -0.926, 1.473, 0.835, 0.534, -1.368],
    #     "MAP-align to HHH-70%": [2.005, -0.869, 1.569, 0.829, 0.546, -1.374],
    #     "MAP-align to HHH-80%": [2.172, -0.934, 1.675, 0.824, 0.550, -1.371],
    # }
    # plot_radar_chart(categories, model_scores=rewards_llama2chat_conversation_MC, title_text='Value Palletes for Llama2-7B-chat Model', output_file="results/fig_radar_llama2chat_conversation_MC.pdf", model_name="llama2_chat")

    # llama2chat Separated into (a) (b)
    rewards_llama2chat_conversation_MC_a = {
        "Original model": [0.604, -1.011, 1.248, 0.848, 0.521, -1.375],
        "MAP to HHH-50%": [0.984, -1.057, 1.362, 0.843, 0.527, -1.367],
        "MAP to HHH-60%": [1.564, -0.926, 1.473, 0.835, 0.534, -1.368],
        "MAP to HHH-70%": [2.005, -0.869, 1.569, 0.829, 0.546, -1.374],
        "MAP to HHH-80%": [2.172, -0.934, 1.675, 0.824, 0.550, -1.371],
    }
    rewards_llama2chat_conversation_MC_b = {
        "Original model": [0.604, -1.011, 1.248, 0.848, 0.521, -1.375],
        "MAP to Humor-80%": [2.208, -1.257, 1.055, 0.815, 0.552, -1.469],
        "MAP to Helpfulness-80%": [0.501, -0.391, 0.941, 0.856, 0.527, -1.362],
        "MAP to Harmlessness-80%": [0.470, -1.328, 1.911, 0.859, 0.501, -1.335],
    }
    plot_radar_chart(categories, model_scores=rewards_llama2chat_conversation_MC_a, title_text='(a) Multi-value Palette Alignment', output_file="results/fig_radar_llama2chat_conversation_MC_a.pdf", model_name="llama2_chat", art_set=1)
    plot_radar_chart(categories, model_scores=rewards_llama2chat_conversation_MC_b, title_text='(b) Single-value Palette Alignment', output_file="results/fig_radar_llama2chat_conversation_MC_b.pdf", model_name="llama2_chat", art_set=2)
