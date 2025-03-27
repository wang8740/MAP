'''
    This script turns the tab results namely the real c levels into quantile representations. 
    But we leave the details out of the paper as they are not perfectly close to the target and can cause reader confusions.
'''

import json
from rewardProcessor import RewardProcessor


# Data extracted from the table for JSON hierarchy
data = {
    "name": "tab_opt13b_Anthropic",
    "50%": {
        "D": [2.437, -1.382, 0.207, 0.884, 0.426, -3.170],
        "F": [2.204, -1.800, 0.655, 0.860, 0.406, -2.801]
    },
    "60%": {
        "D": [2.476, -1.325, 0.481, 0.883, 0.433, -3.149],
        "F": [2.469, -2.255, 0.499, 0.891, 0.257, -3.486]
    },
    "70%": {
        "D": [2.494, -1.287, 0.661, 0.879, 0.446, -3.136],
        "F": [2.174, -2.279, 0.970, 0.823, 0.097, -5.039]
    }
}

# Path to save JSON file
basemodel_name, data_name = "opt1.3b", "Anthropic-harmless" #"llama2_chat", "Anthropic-harmless"

r = RewardProcessor(file_path=f"results/{basemodel_name}-{data_name}.json")
for case in ["50%", "60%", "70%"]:
    data[case]["D-quantile"] = r.quantile_transform_single_c(data[case]["D"])
    data[case]["F-quantile"] = r.quantile_transform_single_c(data[case]["F"])

# Writing JSON data
with open("plot_tab_in_quantiles.json", 'w') as f:
    json.dump(data, f, indent=4)
print("resulted JSON file saved")
