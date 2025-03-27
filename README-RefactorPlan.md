# AlignMAP Refactoring Plan

## Project Overview

AlignMAP (Multi-Human-Value Alignment Palette) is an open-source framework designed to help AI systems better align with diverse human values and preferences. In simpler terms, it provides tools that enable machine learning models, particularly large language models (LLMs)  to generate text that respects and balances multiple human values simultaneously.

### What Problem Does It Solve?

Most AI alignment techniques focus on aligning models with a single value or preference, such as helpfulness or harmlessness. In reality, humans have multiple, sometimes competing values. AlignMAP addresses this challenge by allowing developers to specify multiple values and find the optimal balance between them.

For example, a model might need to balance being truthful, helpful, harmless, and creative all at once. AlignMAP provides the mathematical framework and practical tools to achieve this multi-value alignment.

### Key Features

- **Value Optimization**: Automatically finds the optimal weights (lambda values) to balance multiple human values
- **Reward Model Integration**: Works with various reward models that represent different human values
- **Training Integration**: Supports training methods like PPO (Proximal Policy Optimization) and DPO (Direct Preference Optimization)
- **Inference Time Alignment**: Aligns text generation even without retraining models
- **CLI and API**: Offers both command-line and programmatic interfaces
- **Visualization Tools**: Provides tools to visualize and understand alignment results

### How It Works

AlignMAP works by:
1. Defining multiple reward models that each capture a specific human value (e.g., helpfulness, harmlessness)
2. Finding the optimal mathematical weights to combine these reward models
3. Using these weights to guide the generation of text during inference or to train models

The framework supports both inference-time alignment (where it evaluates multiple candidate outputs and selects the best) and training-time alignment (where it teaches models to directly generate aligned text).

### Who Is It For?

- **AI Researchers**: Studying value alignment and multi-objective optimization
- **ML Engineers**: Building safer and more helpful AI systems
- **AI Safety Practitioners**: Implementing guardrails for AI systems
- **NLP Developers**: Creating more balanced and nuanced text generation

### Project Structure

The project is organized into modular components:
- Core alignment algorithms
- Support for various language models and reward models
- Training utilities for PPO and DPO
- Data loading and processing tools
- Command-line interface for easy usage
- Visualization tools for interpreting results

Contributors can extend any of these components to add new capabilities, models, or alignment techniques.

## Directory Structure

```
alignmap/
├── __init__.py             # Package initialization with version and imports
├── core/                   # Core alignment algorithms
│   ├── __init__.py
│   └── alignment.py        # Core alignment algorithms
├── models/                 # Model definitions and utilities
│   ├── __init__.py
│   ├── language_models/    # Language model implementations
│   │   └── __init__.py
│   ├── reward_models.py    # Main reward model implementation
│   └── reward_models/      # Reward model implementations
│       ├── __init__.py
│       ├── base.py         # Base reward model classes
│       └── registry.py     # Registry for reward models
├── training/               # Training utilities
│   ├── __init__.py
│   ├── ppo.py              # PPO implementation
│   ├── dpo.py              # DPO implementation 
│   └── decoding.py         # Decoding utilities
├── data/                   # Data loading and processing
│   ├── __init__.py
│   └── loaders.py          # Data loaders
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── device.py           # Device utilities
├── cli/                    # Command-line interface
│   ├── __init__.py
│   ├── align.py            # CLI for alignment
│   ├── train.py            # CLI for training
│   └── runners/            # CLI runners
└── visualization/          # Visualization utilities
    └── __init__.py
```

## Implementation Plan

### 1. Core Module

- Refactor `alignValues.py` into `core/alignment.py` for the core alignment algorithms
- Create a clean API for the alignment functions

### 2. Models Module

- Create a base reward model class in `models/reward_models/base.py`
- Implement a registry system for reward models in `models/reward_models/registry.py`
- Move existing reward model implementations to appropriate files
- Add a framework for creating and registering language models

### 3. Training Module

- Refactor `trainPPO.py` into `training/ppo.py`
- Refactor `trainDPO.py` into `training/dpo.py`
- Implement a common interface for training algorithms
- Create a decoding module for generating aligned text

### 4. Data Module

- Create data loaders for common dataset formats
- Implement dataset processing utilities

### 5. Utils Module

- Move common utility functions to appropriate locations
- Create device utilities for managing compute resources

### 6. CLI Module

- Create a clean command-line interface
- Implement subcommands for different operations
- Add proper argument parsing and validation

### 7. Documentation

- Add docstrings to all functions and classes
- Create README.md with usage examples
- Add API documentation

## Additional Files

```
├── scripts/                          # Executable scripts
│   ├── run_alignment.py              # Script to run alignment
│   ├── run_training.py               # Script to run training
│   └── run_eval.py                   # Script to run evaluation
├── examples/                         # Example usage scripts
│   ├── align_values.py               # Example of value alignment
│   ├── train_model.py                # Example of model training
│   └── visualize_results.py          # Example of result visualization
├── docs/                             # Documentation
│   ├── api/                          # API reference
│   ├── tutorials/                    # Tutorials
│   └── examples/                     # Example notebooks
├── tests/                            # Unit tests
│   ├── test_alignment.py
│   ├── test_training.py
│   └── test_utils.py
├── setup.py                          # Package setup
├── pyproject.toml                    # Modern build configuration
├── README.md                         # Project README
└── CONTRIBUTING.md                   # Contributing guidelines
```

## Migration Plan

1. Create the new directory structure
2. Move and refactor code incrementally
3. Implement the core modules first
4. Add tests for each module
5. Update documentation
6. Create example scripts
7. Ensure backward compatibility where possible

## Future Enhancements

1. Add more reward models
2. Improve documentation with tutorials
3. Create a web UI for interactive alignment
4. Add support for more training algorithms
5. Implement visualization tools for alignment results 