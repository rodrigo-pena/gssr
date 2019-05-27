# Graph Signal Sampling and Recovery (GSSR)

[Python][python] functionality for recovering graph signals from subsampled measurements. Currently maintained by [Rodrigo Pena](https://rodrigo-pena.github.io).

Check the [installation guide](#installation) for help setting up your environment to use the tools in the repository.

## Description


## Content


[numerical-tour]: https://nbviewer.jupyter.org/github/rodrigo-pena/gssr/blob/outputs/phd-thesis/numerical-tour.ipynb

## Installation

[![Binder](https://mybinder.org/badge.svg)][binder]
&nbsp; Click the binder badge to run the included notebooks from your browser without installing anything.

[binder]: https://mybinder.org/v2/gh/rodrigo-pena/gssr/master?urlpath=lab

For a local installation, you will need [git], [python >= 3.6][python], [jupyter], and packages from the [python scientific stack][scipy]. I recommend using [conda] for creating and managing a separate environment for this repository. If unfamiliar with this process, you can follow the instructions below.

1. Download the Python 3.x [Miniconda installer][miniconda] and run it using default settings (another option is to download the bulkier [Anaconda installer][anaconda]).
1. Open a terminal window and install git with `conda install git`.
1. Within the terminal, navigate to the directory where you want to keep the contents from this repository (e.g., run `cd ~/Documents/github/`).
1. Download this repository to the current directory by typing `git clone https://github.com/rodrigo-pena/gssr` on the terminal.
1. Create a new environment by typing `conda create --name gssr` on the terminal.
1. Activate the environment with `conda activate gssr` (or `activate gssr`, or `source activate gssr`).
1. You should notice `gssr` typed somewhere on the newest terminal line, indicating that you are within the environment. Install the required packages by running `conda install -c conda-forge python=3.6 jupyterlab`, and then `pip install -r requirements.txt`.
1. Run `python test_install.py` to check if all the requirements have been correctly installed.

After the (one-time) creation of the environment, you can do the following every time you want to work with this repository:

1. Open a terminal and navigate to the directory where the repository has been downloaded.
1. Activate the environment with `conda activate gssr` (or `activate gssr`, or `source activate gssr`).
1. Start Jupyter with `jupyter lab`. The command should open a new tab in your web browser.
1. Edit and run the scripts and notebooks from your browser.

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
[flight_routes]: https://github.com/franckdess/NTDS_Project
[git]: https://git-scm.com
[jupyter]: https://jupyter.org/
[miniconda]: https://conda.io/miniconda.html
[pygsp]: https://github.com/epfl-lts2/pygsp
[python]: https://www.python.org
[pyunlocbox]: https://github.com/epfl-lts2/pyunlocbox
[scipy]: https://www.scipy.org
[swiss_council]: https://github.com/nikolaiorgland/conseil_national
