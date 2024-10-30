
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

Initiate Sphinx:
```bash
sphinx-quickstart docs
```
and choose not to separate build and source

Replace index.rst with index.md if using markdown to generate html. Then, update index.md content as needed.

An example of index.md is
```
# Welcome to MAP Documentation

This is the API documentation for MAP. Enjoy using it!

Author: [Xinran Wang](https://wang8740.github.io)
Contact: wang8740@umn.edu

Github Repo: https://github.com/wang8740/MAP
Documentation: https://wang8740.github.io/MAP/
Paper: https://arxiv.org/abs/2410.19198

If you contribute to the Github [repo](https://github.com/wang8740/MAP) with Google-style comments, the documentation will be automatically updated upon each git push.

~~~{toctree}
:maxdepth: 2

~~~
```



Then, run
```bash
cd docs
sphinx-build -b html . _build
```

The generated HTML files will be available in `docs/_build/`. Use a browser to view the `index.html` file.

If you're working on a headless server, you can:
- Use VS Code Remote - SSH to connect to the server.
- Navigate to `docs/build/html/` and open the HTML files using the HTML Preview plugin.

The File Structure should like like 
```bash
/docs
├── source
│   ├── index.md   # Main entry point for the documentation
│   ├── api.md     # API reference documentation
│   └── conf.py    # Sphinx configuration file
└── build
    └── html       # Generated HTML files
```




## GitHub Pages Deployment

### Set Up GitHub Pages for Hosting Documentation


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

