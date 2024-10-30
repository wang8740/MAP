# MAP (Multi-Human-Value Alignment Palette)

![GitHub stars](https://img.shields.io/github/stars/wang8740/MAP?style=social)
![GitHub forks](https://img.shields.io/github/forks/wang8740/MAP?style=social)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/wang8740/MAP)
![GitHub](https://img.shields.io/github/license/wang8740/MAP)
![GitHub issues](https://img.shields.io/github/issues/wang8740/MAP)
![GitHub pull requests](https://img.shields.io/github/issues-pr/wang8740/MAP)


Artificial Intelligence (AI) has evolved into an integral part of modern technology, affecting many facets of our daily lives and work. Multi-Human-Value Alignment Palette (MAP) offers a first-principle approach to align AI systems with diverse, dynamically defined human values—such as harmlessness, helpfulness, and positiveness—through a structured optimization framework, achieving principled multi-value alignment across tasks.

This repository contains the source codes and API documentation for the MAP project, based on [this paper](https://arxiv.org/pdf/2410.19198).
Citation of the work:
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
- [ ] Packaging
- [ ] Set up use cases on both slurm and notebook
- [ ] Add GUI gif demo 
- [ ] Add github metrics


## Contact

If you have any questions, please feel free to contact us at wang8740@umn.edu or submit an issue.

