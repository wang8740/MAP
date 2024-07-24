# Human-AI Value Alignment

## Use docify plugin in VScode to generate reStructuredText-formatted comments

## Use Sphinx to convert reStructuredText docstrings to Markdown

Generate .rst files from python files
sphinx-apidoc -o source/ .

Build the documentation as Markdown
sphinx-build -b markdown source/ build/

mv build/alignValues.md docs/


## Set up docsify to generate documentation from markdown
install nvm
install Node.js and npm

npm install -g docsify-cli

docsify init ./docs
Edit docs/_sidebar.md to include your generated documentation links:
* [Home](/)
* [API Reference](xxx.md)

docsify serve docs

