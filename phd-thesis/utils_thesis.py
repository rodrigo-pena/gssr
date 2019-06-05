# -*- coding: utf-8 -*-

"""Utilities module for the thesis-related code

"""


import matplotlib as mpl


def set_latex__mpl_preamble():
    r"""
    Set `matplotlib` to typeset LaTeX with specific settings.
    """
    
    mpl.use("pgf")
    pgf_with_custom_preamble = {
        "font.family": "serif", # use serif/main font for text elements
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": [
             "\\usepackage[utf8]{inputenc}",
             "\\usepackage[T1]{fontenc}",
             "\\usepackage{fourier}",         # load additional packages
             "\\usepackage{unicode-math}",  # unicode math setup
             "\\setmainfont{Utopia}", # serif font via preamble
         ]
    }
    mpl.rcParams.update(pgf_with_custom_preamble)