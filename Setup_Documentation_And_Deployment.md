
# Quick Guide on Setting Up API Documentation From Scratch

NOTE: The current repository is already configured to deploy the documentation to the following GitHub Pages automatically upon every push: `https://<your-username>.github.io/<your-repository>/`, e.g., `https://wang8740.github.io/MAP/`

You will only need this guide if you need to 
- rebuild documention on your local environment
- rebuild documentation in a separate remote repo
  
## Prerequisites

To build the documentation from scratch, ensure you have the following Python packages installed:

```bash
pip install sphinx myst-parser sphinx_rtd_theme sphinxcontrib-mermaid sphinx-markdown-builder linkify-it-py sphinx-autoapi
```


### Use Sphinx-autoapi plugin

1. Initiate Sphinx:
```bash
sphinx-quickstart docs
```
and choose not to separate build and source



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


3. Replace `docs/source/index.rst` with `docs/source/index.md` if using markdown to generate html. Then, update this file to include the AutoAPI-generated documentation:
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


4. Run
```bash
cd docs
sphinx-build -b html . _build
```

The File Structure should like like 
```bash
 /docs
├── index.md # Main entry point for the documentation
├── conf.py # Sphinx configuration file
├── make.bat # Batch file for building documentation
├── Makefile # Makefile for building documentation
└── _build # Directory containing generated HTML files
```


The generated HTML files will be available in `docs/_build/index.html`. Use a browser to view the `index.html` file.

If you're working on a headless server, you can use VS Code Remote - SSH to connect to the server, and then open the HTML file using the HTML Preview plugin.


## GitHub Pages Deployment

To trigger automatic refreshing of the documentation upon each git action, use the following template. 

### Set Up GitHub Pages for Hosting Documentation

To automatically deploy your documentation to GitHub Pages with every push to the main branch, follow these steps:

1. Create a `.github/workflows/docs_build_deploy.yml` file in your repository with the following content:

```yaml
name: Deploy GitHub Pages

on:
  push:
    branches: ["main"]

  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
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
        run: pip list
      
      - name: Build MAP's API documentation
        run: |
          cd docs
          sphinx-build -b html . _build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./docs/_build

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```


2. Configure GitHub Pages:
   - Go to your repository's Settings > Pages.
   - Under "Source", select "GitHub Actions" (not "Deploy from a branch").

3. After pushing changes to your main branch, GitHub Actions will automatically build and deploy your documentation.

4. Your documentation will be available at `https://<your-username>.github.io/<your-repository>/`

Note: It may take 1-5 minutes for changes to appear on the GitHub Pages site after a successful build and deployment. If you don't see updates after 5 minutes, check your GitHub Actions logs for any errors.

To manually trigger a rebuild:
- Make a small change to your documentation source in the `main` branch.
- Commit and push this change to trigger the workflow again.

Remember, the `gh-pages` branch is now managed automatically by the GitHub Actions workflow. You don't need to manually update or push to this branch.

