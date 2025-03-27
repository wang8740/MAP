from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="alignmap",
    version="0.1.0",
    author="Jingwen Ding",
    author_email="dingj@example.com",
    description="A framework for aligning language models with human values",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alignmap",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/alignmap/issues",
        "Documentation": "https://github.com/yourusername/alignmap#readme",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "datasets>=2.0.0",
        "trl>=0.4.1",
    ],
    entry_points={
        "console_scripts": [
            "alignmap-train=alignmap.cli.train:main",
            "alignmap-align=alignmap.cli.align:main",
            "alignmap-list-models=alignmap.models.reward_models:list_available_reward_models_cli",
        ],
    },
)
