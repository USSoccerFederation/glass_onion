# Contributing to Glass Onion

<caption><small>This guide is based on Kloppy's [contributing guide](https://kloppy.pysport.org/contributor-guide/contributing/).</small></caption>

When contributing to this repository, please discuss the change you wish to make with the repository owners 
via [Issues](https://github.com/USSoccerFederation/glass_onion/issues) before making the change. This is to ensure that there 
is nobody already working on the same issue and to ensure your time as a contributor isn't wasted!

## How to Contribute

All code changes happen through Pull Requests. If you would like to contribute, follow the steps below to set up 
the project and make changes:

1. Fork the repo and create your branch from `main`.
2. Make code changes to fix a bug or add features.
3. If you have added new code, add test(s) which cover the changes you have made. If you have updated existing code, 
verify that the existing tests cover the changes you have made and add/modify tests if needed. Add docstrings if necessary.
4. Ensure that tests pass using `uv run pytest tests` (or, if you're in VSCode, using the built-in Testing tab).
5. Ensure that your code conforms to the coding standard by executing the command `uv run ruff format` prior to committing your code. 
6. Ensure that any relevant documentation is updated in `docs/` and in docstrings across the project. See [Documentation](#documentation) below.
7. Commit your code and create your Pull Request. Please specify in your Pull Request what change you have made and 
please specify if it relates to any existing issues.  

## Project Setup

After you have forked the code and cloned it to your machine, run `uv sync` to install all dependencies (for the package and for development). `uv` will also maintain a virtual environment for you automatically.

## Code Formatting

This project uses the `ruff` code formatter to ensure all code conforms to a specified format. Please format all of your code using `ruff` prior to committing using `uv run ruff format`.

## Documentation

This project uses [MkDocs](https://www.mkdocs.org/) to generate documentation from pages written in Markdown in `docs/`.

To build docs locally:

```bash
# start MkDocs built-in dev-server
uv run mkdocs serve
```

Open up [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser to preview your documentation. 

This project uses [`mkdocstrings`](https://mkdocstrings.github.io/python/usage/customization/) to auto-generate documentation from classes and functions. Please add/update docstrings where necessary so that relevant documentation also gets added/updated.