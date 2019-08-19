# Graph Signal Sampling and Recovery (GSSR)

[Python][python] functionality for recovering graph signals from subsampled measurements. 

Currently maintained by [Rodrigo Pena](https://rodrigo-pena.github.io).

The [installation guide](#installation) below contains instructions for setting up your environment to use the tools in the repository.



## Description

A **graph** is a collection of vertices and edges. The latter can be weighted, often indicating the degree of similarity between the connected vertices. 

Any numerical quantities attached to the vertices of a graph can be considered as a **graph signal**. For example, in a social network with $n$ vertices, attach a value of $+1$ to all the users who watched [The Social Network][social-network], and $-1$ otherwise. The $n$-dimensional vector gathering the $\pm 1$ values for all the vertices is a graph signal.

Querying each vertex for its signal value can be expensive in large networks, meaning one can usually afford to measure these values only at a few selected locations. This is called **subsampling** and is, mathematically, a linear operation. Let $\mathbf{x} \in \mathbb{R}^{n}$ be a graph signal and construct a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ as follows. If vertex $i$ is sampled, then add a corresponding row to $\mathbf{A}$ filled with zeros, except at position $i$, where it has a $1$. The matrix thus constructed is called the **measurement matrix**, and $\mathbf{x} \mapsto \mathbf{Ax}$ is the subsampling operation. Setting the likelihoods with which each vertex is sampled is called a **sampling design**.

To recover $\mathbf{x}$, the signal of interest, from its measurements $\mathbf{Ax}$, one needs a **decoder**. No decoder can recover *every* graph signal because for any $\mathbf{x}$ there are infinitely many vectors $\mathbf{z} \in \mathbb{R}^{n}$ satisfying the measurement constraints $\mathbf{Az} = \mathbf{Ax}$. However, for some sub-classes of signals within $\mathbb{R}^{n}$, one may find a decoder that almost always suceeds provided the number of measurements is large enough. This idea is fundamental in research areas such as [Compressed Sensing][cs].

This repository contains utilities to conveniently load graphs and signals, and implement sampling designs and decoders.



## Content

- `graphs_signals.py`: Functions related to loading and representing graphs and their signals. See the `data/` folder in this repository for information on the relevant datasets.
- `sampling.py`: Functions related to vertex sampling designs.
- `recovery.py`: Functions related to recovery programs (decoders). 
- `phase_transition.py`: Functions related to the phase transition experiments commonly done in [Compressed Sensing][cs]. They allow one to observe how the recovery error varies with a set of parameters (normally the number of measurements).
- `plotting.py`: Functions related to plotting of graphs and signals. 
- `utils.py`: A miscellaneous collection of helper functions.



## Installation guide

*Remark:*  You can also click the binder badge below to run the included notebooks from your browser without installing anything.

[![Binder](https://mybinder.org/badge.svg)][binder]


For a local installation, you will need [git], [python >= 3.6][python], [jupyter], and packages from the [python scientific stack][scipy]. I recommend using [conda] for creating and managing a separate environment for this repository. If unfamiliar with this process, you can follow the instructions below.

1. Download the Python 3.x [Miniconda installer][miniconda] and run it using default settings (another option is to download the bulkier [Anaconda installer][anaconda]).
1. Open a terminal window and install git with `conda install git`.
1. Within the terminal, navigate to the directory where you want to keep the contents from this repository (e.g., run `cd ~/Documents/github/`).
1. Download this repository to the current directory by typing `git clone https://github.com/rodrigo-pena/gssr` on the terminal.
1. Create a new environment by typing `conda create --name gssr` on the terminal.
1. Activate the environment with `conda activate gssr` (or `activate gssr`, or `source activate gssr`).
1. You should notice `gssr` typed somewhere on the newest terminal line, indicating that you are within the environment. Install the required packages by running `conda install -c conda-forge python=3.6 jupyterlab` and then `pip install -r requirements.txt`.
1. Run `python test_install.py` to check if all the requirements have been correctly installed.



## Using the repository

After the (one-time) creation of the environment, you can do the following every time you want to work with this repository:

1. Open a terminal and navigate to the directory where the repository has been downloaded.
1. Activate the environment with `conda activate gssr` (or `activate gssr`, or `source activate gssr`).
1. Start Jupyter with `jupyter lab`. The command should open a new tab in your web browser.
1. Edit and run the scripts and notebooks from your browser.

You can check `standard_pipeline.ipynb` for a basic idea of how to use the function modules in the repository.



## Acknowledgments

The contents of this repository benefitted from the following resources:

* [PyGSP][pygsp].
* [PyUNLocBoX][pyunlocbox].
* [Voting patterns in the Swiss National Council][swiss_council].
* [Finding Continents from a Flight Routes Network][flight_routes].



## License

The content is released under the terms of the [MIT License](LICENSE.txt).



[anaconda]: https://www.anaconda.com/distribution/
[conda]: https://conda.io
[cs]: https://en.wikipedia.org/wiki/Compressed_sensing
[binder]: https://mybinder.org/v2/gh/rodrigo-pena/gssr/master?urlpath=lab
[flight_routes]: https://github.com/franckdess/NTDS_Project
[git]: https://git-scm.com
[jupyter]: https://jupyter.org/
[miniconda]: https://conda.io/miniconda.html
[pygsp]: https://github.com/epfl-lts2/pygsp
[python]: https://www.python.org
[pyunlocbox]: https://github.com/epfl-lts2/pyunlocbox
[scipy]: https://www.scipy.org
[social-network]: https://www.imdb.com/title/tt1285016/
[swiss_council]: https://github.com/nikolaiorgland/conseil_national 
