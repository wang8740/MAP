# MAP (Multi-Human-Value Alignment Palette)

![GitHub stars](https://img.shields.io/github/stars/wang8740/MAP?style=social)
![GitHub forks](https://img.shields.io/github/forks/wang8740/MAP?style=social)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/wang8740/MAP)
![GitHub](https://img.shields.io/github/license/wang8740/MAP)
![GitHub issues](https://img.shields.io/github/issues/wang8740/MAP)
![GitHub pull requests](https://img.shields.io/github/issues-pr/wang8740/MAP)

Multi-Human-Value Alignment Palette (MAP) offers a first-principle approach to align AI systems with diverse, dynamically defined human values—such as harmlessness, helpfulness, and positiveness—through a structured optimization framework, achieving principled multi-value alignment across tasks.

This repository contains the source code and API documentation for the MAP project, based on [this paper](https://arxiv.org/pdf/2410.19198).

## Installation

You can install MAP directly from PyPI:

```bash
pip install alignmap
```

Or install from source:

```bash
git clone https://github.com/wang8740/MAP.git
cd MAP
pip install -e .
```

## Key Features

- **Core Alignment Algorithm**: Optimize lambda coefficients to combine multiple reward models based on target palettes
- **Extensible Reward Model Framework**: Use built-in reward models or easily add your own
- **Training Support**: Includes PPO and DPO implementations for model alignment
- **GPU Optimization**: Automatic device detection and resource allocation
- **CLI Tools**: Command-line interface for training and alignment
- **Cluster Support**: Submit jobs to PBS clusters with customizable templates

## Quick Start

### Basic Alignment

```python
from alignmap import align_values
import torch

# Define rewards for 3 values across 100 samples
rewards = torch.randn(3, 100)

# Define target palette (desired improvements for each value)
target_palette = [0.5, 0.8, 0.3]

# Align the values
lambda_values, success = align_values(
    values=["helpfulness", "harmlessness", "honesty"],
    rewards=rewards,
    target_palette=target_palette,
    verbose=True
)

print(f"Optimized lambda values: {lambda_values}")
```

### Creating Custom Reward Models

```python
from alignmap.models.reward_models import BaseRewardModel, register_reward_model

@register_reward_model("my_custom_reward")
class MyRewardModel(BaseRewardModel):
    def __init__(self, device=None):
        super().__init__("my_custom_reward", device)
        # Initialize your model
        
    def calculate_reward(self, texts, prompts=None, **kwargs):
        # Calculate rewards for texts
        rewards = [...]  # Your reward calculation logic
        return rewards
```

### Training with PPO

```python
# Command-line usage
alignmap-train --model gpt2 --method ppo --reward-model helpfulness harmlessness --dataset Anthropic-harmless --learning-rate 1e-6
```

### Using GPU

```python
from alignmap.utils import get_device, set_device_settings

# Automatically detect and configure device
device = get_device()
set_device_settings(device, mixed_precision=True, memory_efficient=True)

# Use in alignment
lambda_values, success = align_values(
    values=["helpfulness", "harmlessness"],
    rewards=rewards,
    target_palette=[0.5, 0.3],
    device=device
)
```

### Submitting to Compute Cluster

```python
# Command-line usage
alignmap-train --model llama-7b --method ppo --reward-model helpfulness harmlessness --runner pbs --template my_template.pbs
```

## Documentation

For full documentation, visit [https://wang8740.github.io/MAP/](https://wang8740.github.io/MAP/).

## Citation

If you use MAP in your research, please cite:

```
@misc{wang2024mapmultihumanvaluealignmentpalette,
      title={MAP: Multi-Human-Value Alignment Palette}, 
      author={Xinran Wang and Qi Le and Ammar Ahmed and Enmao Diao and Yi Zhou and Nathalie Baracaldo and Jie Ding and Ali Anwar},
      year={2024},
      eprint={2410.19198},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.19198}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Quick Start

Run the following to install the package from [Pypi](https://pypi.org/project/alignMAP/1.0.0/)
```bash
pip install alignMAP==1.0.0
```

### Submitting Jobs with alignMAP

The following example will help you submit jobs to a SLURM server using the alignMAP package. It guides you through generating text data, calculating reward scores, and assessing model alignment.

Create a `submit.py' that contains:
```python
import os
import random
from alignMAP.utils import ALL_SUPPORTED_VALUES, convert_ppo_modelname_to_huggingface_valid
from alignMAP import trainDPO, gendata, rewardProcessor, mergeProcessor

def random_string(length=8):
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(length))

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Parameters
basemodel_name = "opt1.3b"
sample_size = 2000
beta = 0.5
harmless_ratio = 0.9
gen_data_name = "Anthropic-harmless"

DPO_model_name = f"{basemodel_name}-{sample_size}sample-{beta}beta-{harmless_ratio}harmless"
dpoModel_relative_path = f"modelsDPO/{DPO_model_name}"
dpoModel_relative_path = convert_ppo_modelname_to_huggingface_valid(dpoModel_relative_path)
dpoModel_abs_path = os.path.abspath(dpoModel_relative_path)
json_filepath = f"{dpoModel_relative_path}-{gen_data_name}.json"

# Commands for job submission
commands = [
    f'python trainDPO.py --sample_size={sample_size} --beta={beta} --harmless_ratio={harmless_ratio} --save_path={dpoModel_relative_path}',
    f'python gendata.py --basemodel_name="{dpoModel_abs_path}" --data_name="{gen_data_name}" --save_directory="modelsDPO" generate_from_original_model',
] + [
    f'python rewardProcessor.py --value="{value}" --file_path={json_filepath} --basemodel_for_perplexity={dpoModel_abs_path} add_reward' 
    for value in ALL_SUPPORTED_VALUES
] + [
    f'python mergeProcessor.py --original_file_path={json_filepath} merge_added_rewards',
    f'python rewardProcessor.py --file_path={json_filepath} --values_to_evaluate="all" --evaluation_mode=True assess_original_value',
    f'rm -rf {dpoModel_relative_path}'
]

# PBS Job Setup
template_path = 'main.pbs'  # adjust for your SLURM setup
jobs_dir = 'pbs-files'
ensure_dir(jobs_dir)

# Load PBS Template
with open(template_path, 'r') as template_file:
    pbs_content = template_file.read()

# Replace placeholder with commands
pbs_content = pbs_content.replace("COMMAND_PLACEHOLDER", "\n".join(commands))

# Save job file
job_file_name = os.path.join(jobs_dir, f'job_{random_string()}.pbs')
with open(job_file_name, 'w') as job_file:
    job_file.write(pbs_content)
print(f'Created job file {job_file_name}')

# Submit job
os.system(f'sbatch {job_file_name}')
```

and also create a pre-configured PBS template file [`main.pbs`](examples/main.pbs).

Then, run `python submit.py` and wait for the email notifications of results.

## Make Contributions to the Repo

The current repo strcuture:
```
MAP/
├── alignMAP/                    # Source directory for your package modules
│   ├── alignValues.py
│   ├── gendata.py
│   ├── gendataGUI.py
│   └── ...                      # Other Python modules
├── docs/                        # Documentation
├── .github/workflows/           # GitHub workflows
├── tests/                       # Tests for your package
│   ├── test_alignValues.py      # Test files (example)
│   └── ...
├── MAP/                         # Package directory (new)
│   ├── __init__.py              # Marks this directory as a package
│   └── other package files
├── README.md                    # Project README
├── setup.py                     # Package configuration
├── requirements.txt             # Package dependencies
└── .gitignore                   # Ignored files and directories
```

The current repository is configured to deploy/refresh the documentation to the following GitHub Pages `https://wang8740.github.io/MAP/` automatically upon every push
To contribute, please follow the following steps.

### Use Google-Style Comments

Install VS Code Extensions: 
- `AutoDocstring` that generates docstrings automatically in Google style
- `Python Type Hint` that assist in adding typing for each arugment 

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Example function summary.

    Args:
        param1 (int): Description of param1.
        param2 (str): Description of param2.

    Returns:
        bool: Description of return value.
    """
    return True
```


### Contributing via git push to remote 

Commit and push the changes:
```bash
git add .
git commit -m "Update documentation"
git push origin main
```

Once you push, the documentation will be automatically refreshed in about one minute. 
If you need to rebuild the documentation locally or set up the Github workflow from scratch, please refer to [Setup_Documentation_And_Deployment.md](Setup_Documentation_And_Deployment.md).


### Contributing via pull requests

We welcome contributions from the community! If you'd like to contribute to this project, please follow these steps:

1. **Fork the Repository**
   
   Click the "Fork" button at the top right of this repository's page on GitHub to create your own copy.

2. **Clone Your Fork**
   
   Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name
   ```

3. Create a New Branch Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. Commit Your Changes
   ```bash
   git add .
   git commit -m "Add feature: brief description of your changes"
   ```
   
5. Push to Your Fork
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

- Go to the original repo on GitHub
- Click on "Pull requests" and then the "New pull request" button
- Choose your fork and the branch you created
- Click "Create pull request"
- Provide a title and description for your pull request

7. Wait for Review The project maintainers will review your pull request. They may ask for changes or clarifications.

8. Once approved, a project maintainer will merge your pull request.


## Summary of Recent Papers on Multi-Objective Alignment
For convenience, we also list of recent papers focused on multi-objective alignment. While this list is not exhaustive, we encourage contributions to enrich it further. Feel free to add any relevant papers or resources that you believe would benefit the community.

| Title | Author(s) | Year | URL |
|-------|-----------|------|-----|
| Rewards-in-context: Multi-objective alignment of foundation models with dynamic preference adjustment | Rui Yang, Xiaoman Pan, Feng Luo, Shuang Qiu, Han Zhong, Dong Yu, Jianshu Chen | ICML 2024 | [Link](https://arxiv.org/abs/2403.12805) |
| Safe RLHF: Safe reinforcement learning from human feedback | Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, Yaodong Yang | ICLR 2024 | [Link](https://arxiv.org/abs/2310.12773) |
| Fine-grained human feedback gives better rewards for language model training | Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A Smith, Mari Ostendorf, Hannaneh Hajishirzi | NeurIPS 2024 | [Link](https://arxiv.org/abs/2306.01693) |
| Contextual moral value alignment through context-based aggregation | Pierre Dognin, Jesus Rios, Ronny Luss, Inkit Padhi, Matthew D Riemer, Miao Liu, Prasanna Sattigeri, Manish Nagireddy, Kush R Varshney, Djallel Bouneffouf | 2024 | [Link](https://arxiv.org/abs/2403.12805) |
| Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards | Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, Matthieu Cord | NeurIPS 2023 | [Link](https://arxiv.org/abs/2306.04488) |
| Training a helpful and harmless assistant with reinforcement learning from human feedback | Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan | 2022 | [Link](https://arxiv.org/abs/2204.05862) |


## How to Contribute

We welcome contributions from everyone. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.


## Code of Conduct

To ensure a welcoming and productive environment, all participants are expected to uphold our [Code of Conduct](./CODE_OF_CONDUCT.md).


## To-Do list 
- [x] Packaging
- [x] Set up use cases on both slurm and notebook
- [ ] Add GUI gif demo 
- [x] Add github metrics
- [x] Refactor codebase to follow best practices
- [x] Add comprehensive tests for core functionality
- [x] Add example scripts demonstrating usage
- [x] Create CLI for alignment and training


## Refactoring Status

The codebase has been refactored from the original structure in `backend/` to a more modular organization:

```
alignmap/
├── core/                   # Core alignment algorithms
├── models/                 # Model definitions and adapters
│   ├── language_models/    # Language model adapters
│   └── reward_models/      # Reward model implementations
├── training/               # Training utilities for PPO/DPO
├── data/                   # Data loading and processing
├── utils/                  # Utility functions
├── cli/                    # Command-line interfaces
└── visualization/          # Visualization utilities
```

Key improvements include:

1. **Modular Architecture**: Code is now organized by functionality
2. **Clean API**: Clear interfaces for each component
3. **Type Hints**: Added comprehensive type annotations
4. **Documentation**: Google-style docstrings for all functions
5. **Testing**: Unit and integration tests
6. **Examples**: Comprehensive examples showing how to use the framework

The refactoring has maintained all functionality from the original codebase while adding:
- Better language model support through the adapter pattern
- Cleaner reward model registry system
- More robust error handling
- Improved device management
- More efficient batch processing


## Contact

If you have any questions, please feel free to contact us at wang8740@umn.edu or submit an issue.

## Environment Variables

AlignMAP uses environment variables for configuration. To set up your environment:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and fill in your values:
   ```bash
   # Replace with your actual OpenAI API key
   OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. The environment variables will be automatically loaded when running the application.

