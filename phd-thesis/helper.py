# -*- coding: utf-8 -*-

"""Utilities module for the thesis-related code

"""


import matplotlib as mpl


def set_pgf_preamble():
    r"""
    Set `matplotlib` to typeset LaTeX with specific settings.
    """
    
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
    
    pgf_setup = {
        "pgf.rcfonts": False,                 # setup fonts from rc parameters
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": [
             "\\usepackage[T1]{fontenc}",
             "\\usepackage[utf8]{inputenc}",
             "\\usepackage{fourier}"  
         ]
    }
    
    mpl.rcParams.update(pgf_setup)
    
def vanish_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
        
def vanish_ticks(ax, axis='both'):
    ax.tick_params(axis=axis, length=0, labelsize=0)