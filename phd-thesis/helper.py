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


def figsize_from_textwidth(scale):
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
    
    #mpl.use('pgf')
    
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
    
    pgf_setup = {
        "font.serif": "serif",
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 11,               # LaTeX default is 10pt font.
        "font.size": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.figsize": figsize_from_textwidth(0.9),   
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


def plot_colorbar(ax=None, pos=[0.0, 0.0, 0.05, 1.0],
                  vmin=0.0, vmax=1.0, cmap=mpl.cm.Reds, **kwargs):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes(pos)
        
    cb = mpl.colorbar.ColorbarBase(ax=ax, cmap=cmap, norm=norm, **kwargs)
    
    return cb


def draw_arrow_text(target_coords, text_coords, text_str, ax=None):
    x_target = target_coords[0]
    y_target = target_coords[1]
    
    x_text = text_coords[0]
    y_text = text_coords[1]
    
    x_arrow_increment = x_target - x_text
    y_arrow_increment = y_target - y_text
    
    if ax is None:
        fig, ax = plt.subplots(ncols=1) 

    ax.arrow(x_text, 
             y_text, 
             x_arrow_increment, 
             y_arrow_increment, 
             linewidth=0.5, 
             overhang=0.5,
             width=0.0, 
             head_width=1, 
             head_length=1,
             facecolor='k',
             edgecolor='k')

    ax.text(x_text, 
            y_text, 
            text_str,
            ha='left', 
            va='bottom', 
            size=8, 
            color='k',
            bbox=dict(facecolor='w', edgecolor='w', alpha=0.0, pad=0.0),
            rotation=0)


def indicate_ssbm_thresholds(experiment, ax, text_height=3, text_width=23):
    
    y_top = len(experiment['rows']) 
    y_mid = len(experiment['rows']) / 2
    y_bottom = 0.0

    x_right = len(experiment['rows'])
    x_left = 0.0

    text_height = 3 # In pixels
    text_width = 23 # In pixels

    target_coords = (x_left + 0.7, y_mid)
    text_coords = (x_right - text_width, 2 * y_mid - text_height)
    draw_arrow_text(target_coords, text_coords, "Exact recovery threshold", ax=ax)

    target_coords = (x_left + 0.7, y_bottom + 0.5)
    text_coords = (x_right - text_width, y_mid - text_height)
    
    draw_arrow_text(target_coords, text_coords, "Connectivity threshold", ax=ax)
    
    
def plot_sbm_pt(experiment, ax=None, with_colorbar=False, with_thresholds=False):
    r"""Plot SBM phase transition experiment."""
    
    epfl_colors = utils.load_obj('epfl_colors_hex.pkl')
    
    fig = None
    
    if ax is None:
        fig, ax = plt.subplots(ncols=1) 
    else:
        fig = plt.gcf()
        
    im = ax.imshow(experiment['grid'], vmin=0, vmax=1)
    
    if with_colorbar:
        cb = add_colorbar(im, ax)

    x = experiment['cols'].astype(int)
    
    ax.grid(False)
    #vanish_spines(ax)
    
    
    ax.set_xlabel('\# Measurements (m)')
    ax.xaxis.set_major_locator(ticker.LinearLocator(3))
    ax.set_xticklabels(["{0}".format(x[0]),
                        "{0}".format(int(np.median(x))),
                        "{0}".format(x[-1])])

    y = experiment['rows']
    
    ax.set_ylabel(r'Connection probability factor ($a$)')
    ax.yaxis.set_major_locator(ticker.LinearLocator(3))
    ax.set_yticklabels(["{0:.1f}".format(y[0]),
                        "{0:.1f}".format(np.median(y)),
                        "{0:.1f}".format(y[-1])])
    
    if with_thresholds:
        indicate_ssbm_thresholds(experiment, ax=ax)

    if with_colorbar:
        return fig, ax, cb
    else:
        return fig, ax
    
    
def plot_line_pt(experiment, ax=None, label='', **kwargs):
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = plt.gcf()

    ax.plot(experiment['cols'], 
            experiment['line'], 
            label=label,  
            **kwargs)

    ax.set_ylim(bottom=-0.1, top=1.1)

    ax.set(aspect='auto',
           xlabel=r'\# Measurements ($m$)', 
           ylabel=r'Relative $\ell_2$ error')
    
    
def set_cmap(choice='default'):
    
    from matplotlib import rcParams
    
    if choice == 'default':
        rcParams['image.cmap'] = 'Reds'
    elif choice == 'signal':
        rcParams['image.cmap'] = 'cividis'
    elif choice == 'quantized':
        rcParams['image.cmap'] = 'tab10'
    else:
        raise ValueError("Possible colormap choices are {'default', 'signal', 'quantized'}")
    
