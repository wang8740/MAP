# Human-AI Value Alignment

## Use docify plugin in VScode to generate reStructuredText-formatted comments

## Use Sphinx to convert reStructuredText docstrings to Markdown

Generate .rst files from python files
sphinx-apidoc -o source/ .

Build the documentation as Markdown
sphinx-build -b markdown source/ build/

mv build/* docs/


## Set up docsify to generate documentation from formatted code
Install nvm
Install Node.js and npm

docsify init ./docs
Edit docs/_sidebar.md to include your generated documentation links:
* [Home](/)
* [API Reference](api.md)

docsify serve docs
