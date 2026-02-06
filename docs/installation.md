# Installation

<caption><small>This guide is based on Kloppy's [installation guide](https://kloppy.pysport.org/user-guide/installation/).</small></caption>

Before you can use Glass Onion, you'll need to get it installed. This guide will guide you to a minimal installation that'll work while you walk through the user guide.

## Install Python

Being a Python library, Glass Onion requires Python. Currently, Glass Onion supports Python version 3.11+. Get the latest version of Python at [python.org](https://www.python.org/downloads/) or with your operating system's package manager.

You can verify that Python is installed by typing `python` from your shell; you should see something like:

```
Python 3.x.y
[GCC 4.x] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

## Install Glass Onion

You've got three options to install Glass Onion.

### Installing an official release with `uv`

This is the recommended way to install Glass Onion. Simply run this simple command in your terminal of choice:

```console
$ uv add glass_onion
```

You might have to install uv first. See installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).


### Installing an official release with `pip`

Simply run this simple command in your terminal of choice:

```console
$ python -m pip install glass_onion
```

You might have to install pip first. The easiest method is to use the [standalone pip installer](https://pip.pypa.io/en/latest/installation/).

### Installing the development version

Glass Onion is actively developed on GitHub, where the code is [always available](https://github.com/USSoccerFederation/glass_onion). You can easily install the development version with:

```console
$ pip install git+https://github.com/USSoccerFederation/glass_onion.git
```

However, to be able to make modifications in the code, you should either clone the public repository:

```console
$ git clone git://github.com/USSoccerFederation/glass_onion.git
```

Or, download the [zipball](https://github.com/USSoccerFederation/glass_onion/archive/master.zip):

```console
$ curl -OL https://github.com/USSoccerFederation/glass_onion/archive/master.zip
```

Once you have a copy of the source, you can embed it in your own Python package, or install it into your site-packages easily:

```console
$ cd glass_onion
$ python -m pip install -e .
```

## Verifying

To verify that Glass Onion can be seen by Python, type `python` from your shell. Then at the Python prompt, try to import it:

```python
>>> import glass_onion
>>> print(glass_onion.__version__)
```