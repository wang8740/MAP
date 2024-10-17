# MAP: Human-AI Value Alignment

This repository contains the source codes and API documentation for the MAP project.





## Prerequisites

To build the documentation locally, ensure you have the following Python packages installed:

```bash
pip install pydoc-markdown sphinx myst-parser sphinx_rtd_theme sphinxcontrib-mermaid sphinx-markdown-builder linkify-it-py
```

### Building the Documentation Locally
```bash
cd docs
make html
```

The generated HTML files will be available in `docs/build/html/`. Use a browser to view the `index.html` file.

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

### Convert Code to Markdown Files

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

### (Optionally) Re-create the Documentation under docs/

If you want to deploy on another remote repo, where sphinx is not initiated, use the following steps

Under `docs/` initiate Sphinx:
```bash
sphinx-quickstart
```

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

The current repository is configured to deploy the documentation to the following GitHub Pages automatically upon every push
```php
https://<your-username>.github.io/<your-repository>/
```

### (Optional) Set Up GitHub Pages for Hosting Documentation

If you want to deploy on another remote repo, use GitHub Actions to automate the process of updating documentation with every push to GitHub. 

- Ensure you push the build/html directory to a gh-pages branch on your GitHub repository

- In your project’s GitHub repository, create a `.github/workflows/pages.yml` file to automate documentation deployment:

```yaml
name: Deploy Sphinx Docs to GitHub Pages

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.x'
      - name: Install dependencies
        run: |
          pip install sphinx myst-parser sphinx_rtd_theme sphinxcontrib-mermaid
      - name: Build docs
        run: |
          cd docs && make html
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/build/html
```

Enable GitHub Pages:

Go to your repository’s Settings > Pages.
Select the gh-pages branch as the source for your GitHub Pages.

The site will be available at https://<your-username>.github.io/<your-repository>/, e.g., https://wang8740.github.io/MAP/