# MAP: Human-AI Value Alignment

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

## Prerequisites

To build the documentation locally, ensure you have the following Python packages installed:

```bash
pip install sphinx myst-parser sphinx_rtd_theme sphinxcontrib-mermaid sphinx-markdown-builder linkify-it-py sphinx-autoapi
# pip install pydoc-markdown 
```

### Building the Documentation Locally
```bash
cd docs
sphinx-build -b html . _build
```

The generated HTML files will be available in `docs/_build/`. Use a browser to view the `index.html` file.

If you're working on a headless MSI server, you can:
- Use VS Code Remote - SSH to connect to the server.
- Navigate to `docs/build/html/` and open the HTML files using the HTML Preview plugin.

### File Structure
```bash
/docs
├── source
│   ├── index.md   # Main entry point for the documentation
│   ├── api.md     # API reference documentation
│   └── conf.py    # Sphinx configuration file
└── build
    └── html       # Generated HTML files
```



## Make Contributions to the Codes

### Follow Google-Style Docstrings

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

### (Depreciated) Convert Code to Markdown Files Using Pydoc 

Create a pydoc-markdown configuration file `pydoc-markdown.yml` in your project root:
```yaml
loaders:
  - type: python
    search_path: ["./"]

renderer:
  type: markdown
  use_fixed_header_levels: true
  header_level_by_type:
    module: 2
    class: 3
    function: 4
  filename: "docs/source/api.md"
```

Run the following to auto-generate markdown documentation:
```bash
pydoc-markdown
```

The above will generate `docs/source/api.md`

If you clone this repo, the sphinx has been initated, so simply re-build the documentation:
```bash
cd docs
make html
```

### Use sphinx-autoapi

### (Optionally) Re-create the Documentation under docs/

If you want to deploy on another remote repo, where sphinx is not initiated, use the following steps

Initiate Sphinx:
```bash
sphinx-quickstart docs
```
and choose not to separate build and source

Replace index.rst with index.md if using markdown to generate html

Then, update index.md content as needed


### Push to remote

Commit and push the changes:
```bash
git add .
git commit -m "Update documentation"
git push origin main
```



## GitHub Pages Deployment

The current repository is configured to deploy the documentation to the following GitHub Pages automatically upon every push: `https://<your-username>.github.io/<your-repository>/`, e.g., `https://wang8740.github.io/MAP/`


### (Optional) Set Up GitHub Pages for Hosting Documentation


### Set Up GitHub Pages for Hosting Documentation

To automatically deploy your documentation to GitHub Pages with every push to the main branch, follow these steps:

1. Create a `.github/workflows/docs_build_deploy.yml` file in your repository with the following content:

```yaml
name: Build and Deploy Documentation

on:
  push:
    branches:
      - main  # or your default branch name

# Add this permissions block
permissions:
  contents: write
  
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.x'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install sphinx myst-parser sphinx_rtd_theme sphinxcontrib-mermaid sphinx-markdown-builder linkify-it-py sphinx-autoapi

    - name: List installed packages
      run: |
        pip list
    
    - name: Build MAP's API documentation
      run: |
        cd docs
        sphinx-build -b html . _build

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/build/html
        publish_branch: gh-pages  # Specify the branch explicitly
```

2. Ensure your docs/source/conf.py file is correctly configured to use AutoAPI:
```python
extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension',
    ...
]

autoapi_type = 'python'
autoapi_dirs = ['../../']  # Adjust this path to point to your Python source code
autoapi_add_toctree_entry = True
```

3. Update your `docs/source/index.md` file to include the AutoAPI-generated documentation:
~~~
# Welcome to MAP Documentation

Author: [Xinran Wang](https://wang8740.github.io)

Contact: wang8740@umn.edu

This is the API documentation for MAP. Enjoy using it!

```{toctree}
:maxdepth: 2
:caption: Contents:

autoapi/index
```
~~~

4. Configure GitHub Pages:
   - Go to your repository's Settings > Pages.
   - Under "Source", select "GitHub Actions" (not "Deploy from a branch").

5. After pushing changes to your main branch, GitHub Actions will automatically build and deploy your documentation.

6. Your documentation will be available at `https://<your-username>.github.io/<your-repository>/`

Note: It may take 5-15 minutes for changes to appear on the GitHub Pages site after a successful build and deployment. If you don't see updates after 30 minutes, check your GitHub Actions logs for any errors.

To manually trigger a rebuild:
- Make a small change to your documentation source in the `main` branch.
- Commit and push this change to trigger the workflow again.

Remember, the `gh-pages` branch is now managed automatically by the GitHub Actions workflow. You don't need to manually update or push to this branch.


## How to Contribute

We welcome contributions from everyone. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.


## Code of Conduct

To ensure a welcoming and productive environment, all participants are expected to uphold our [Code of Conduct](./CODE_OF_CONDUCT.md).


