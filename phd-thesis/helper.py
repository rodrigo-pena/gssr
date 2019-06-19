# -*- coding: utf-8 -*-

"""Utilities module for the thesis-related code

"""

import utils

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

mpl.use('pgf')

def figsize(scale):
    r""" Figure scaled according to default \textwidth.
    
    Notes
    -----
    Source : http://bkanuka.com/posts/native-latex-plots/
    
    """
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


def set_pgf_preamble():
    r"""
    Set `matplotlib` to typeset LaTeX with specific settings.
    """
    
    #from matplotlib.backends.backend_pgf import FigureCanvasPgf
    #mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
    
    pgf_setup = {
        "font.serif": "serif",
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 11,               # LaTeX default is 10pt font.
        "font.size": 11,
        "legend.fontsize": 11,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.figsize": figsize(0.9),   
        "text.usetex": True,
        "pgf.rcfonts": False,                 # Disable setting up fonts from rcParams
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": [
             "\\usepackage[utf8x]{inputenc}",
             "\\usepackage[T1]{fontenc}",
             "\\usepackage{fourier}", 
         ]
    }
    
    mpl.rcParams.update(pgf_setup)
    
def vanish_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    
def vanish_ticks(ax, axis='both'):
    ax.tick_params(axis=axis, length=0, labelsize=0)
    
    
def add_colorbar(im, ax, position='right'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="5%", pad=0.05)
    return colorbar(im, cax=cax, ticks=mpl.ticker.MaxNLocator(nbins=2))
    
    
def plot_sbm_pt(experiment, ax=None):
    r"""Plot SBM phase transition experiment."""
    
    epfl_colors = utils.load_obj('epfl_colors_hex.pkl')
    
    fig = None
    
    if ax is None:
        fig, ax = plt.subplots(ncols=1) 
    else:
        fig = plt.gcf()
        
    im = ax.imshow(experiment['grid'], vmin=0, vmax=1)
    
    cb = add_colorbar(im, ax)
    vanish_spines(cb.ax)
    cb.ax.tick_params(axis='both', length=0)
    
    connect_thresh = ax.axhline(y=-0.3, 
                                xmin=0, 
                                xmax=1, 
                                linestyle='solid', 
                                linewidth=2,
                                color=epfl_colors['leman'], 
                                dash_capstyle='butt')

    exact_thresh = ax.axhline(y=len(experiment['rows'])/2 - .5, 
                              xmin=0, 
                              xmax=1, 
                              linestyle='solid', 
                              linewidth=2, 
                              color=epfl_colors['canard'], 
                              dash_capstyle='butt')

    leg = ax.legend(handles=[exact_thresh, connect_thresh], 
                    labels=["Exact recovery threshold", "Connectivity threshold"], 
                    frameon=False, 
                    facecolor='inherit',
                    loc='upper right',
                    bbox_to_anchor=(0.76, 1.16), 
                    borderaxespad=0.)

    x = experiment['cols'].astype(int)
    
    ax.grid(False)
    vanish_spines(ax)
    
    
    ax.set_xlabel('\# Measurements (m)')
    ax.xaxis.set_major_locator(ticker.LinearLocator(3))
    ax.set_xticklabels(["{0}".format(x[0]),
                        "{0}".format(int(np.median(x))),
                        "{0}".format(x[-1])])

    y = experiment['rows']
    
    ax.set_ylabel('Connection probability factor (a)')
    ax.yaxis.set_major_locator(ticker.LinearLocator(3))
    ax.set_yticklabels(["{0:.1f}".format(y[0]),
                        "{0:.1f}".format(np.median(y)),
                        "{0:.1f}".format(y[-1])])

    return fig, ax, cb, leg, exact_thresh, connect_thresh    