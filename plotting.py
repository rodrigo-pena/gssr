#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plotting module

"""


import datetime

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


def pt_grid(grid, add_cbar=True, vmin=None, vmax=None, save_to_disk=False, 
            save_dir='data/', file_name='pt_grid'):
    r"""
    Plot in the style of a phase transition grid.
    
    Parameters
    ----------
    grid : array_like
        The pixel grid to plot.
    add_cbar : bool, optional
        Include colorbar on plot. (default is True)
    vmin : float, optional
        Minimum value in `matplotlib.pyplot.imshow`.
    vmax : float, optional
        Maximum value in `matplotlib.pyplot.imshow`.
    save_to_disk : bool, optional
        Save plot to disk. (default is False)
    save_dir : str, optional
        Directory onto which save the plot. (default is 'data/')
    file_name : str, optional
        File name. (default is 'pt_grid')
    
    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Top level container for the plot element.
    
    ax : :class:`matplotlib.axes.Axes`
        Container for figure elements. Sets the coordinate system.

    """
    
    # Global settings
    figsize = (8.3, 8.3) # Width (in inches) of an A4 paper
    dpi = 300            # Good for printing

    fig, ax = plt.subplots(figsize=figsize, ncols=1) 
    im = ax.imshow(grid, 
                   cmap='Reds', 
                   interpolation='none', 
                   origin='lower', 
                   vmin=vmin, 
                   vmax=vmax)
    
    if add_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05) # Fix height
        cb = colorbar(im, cax=cax, ticks=mpl.ticker.MaxNLocator(nbins=2))
        
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
        
    if save_to_disk:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        save_path = save_dir + now + ' ' + file_name + '.pdf'
        plt.savefig(save_path, dpi=dpi, transparent=True)
        
    return fig, ax


def pt_grid_experiment(experiment, file_name='pt_grid_experiment', **kwargs):
    r"""
    Plot in the style of a phase transition grid.
    
    Parameters
    ----------
    experiment : dict
        A dictionary with the results of an experiment like the one returned 
        by :func:`phase_transition.grid_evaluation()`.
    file_name : str, optional
        File name. (default is 'pt_grid_experiment')
    **kwargs
        See :func:`pt_grid()`.
    
    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Top level container for the plot element.
    
    ax : :class:`matplotlib.axes.Axes`
        Container for figure elements. Sets the coordinate system.
    
    """
    
    try:
        file_name = experiment['file_name']
    except KeyError:
        pass
    
    fig, ax = pt_grid(experiment['grid'], file_name=file_name, **kwargs)
    
    try:
        ax.set_xlabel(experiment['col_label'])
    except KeyError:
        pass
     
    try:
        ax.set_ylabel(experiment['row_label'])
    except KeyError:
        pass
    
    try:
        ticklabels = experiment['col_tick_labels']
        n_ticks = len(ticklabels)
        ax.xaxis.set_major_locator(plt.LinearLocator(numticks=n_ticks))
        ax.set_xticklabels(ticklabels)
    except KeyError:
        pass
    
    try:
        ticklabels = experiment['row_tick_labels']
        n_ticks = len(ticklabels)
        ax.yaxis.set_major_locator(plt.LinearLocator(numticks=n_ticks))
        ax.set_yticklabels(ticklabels)
    except KeyError:
        pass
    
    ax.tick_params()
    
    return fig, ax
    
    

def scatter_council(coords, path='data/swiss-national-council-50/', colors=None, 
                    save_to_disk=False, save_dir='data/', file_name='swiss council'):
    r"""
    Scatter plot of the Swiss National Council members.

    Paameters
    ---------
    coords : (N, 2) `numpy.ndarray`
        The first column lists the horizontal coordinates for each council member in the 
        scatter plot. The second column lists the vertical coordinates.
    path : str, optional
        The path to the directory containing the file `council_info.csv`.
    colors : array_like, optional 
        A list specifying a color (or position within a colormap) for each council member. 
        (default is given by the column 'Color' in `council_df`)
    save_to_disk : bool, optional
        Save plot to disk. (default is False)
    save_dir : str, optional
        Directory onto which save the plot. (default is 'data/')
    file_name : str, optional
        File name. (default is 'pt grid')
        
    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Top level container for the plot element.
    
    ax : :class:`matplotlib.axes.Axes`
        Container for figure elements. Sets the coordinate system.
        
    """
    
    # Global settings
    fontsize = 'x-large'
    dpi = 300
    
    fig, ax = plt.subplots(figsize=(8, 8), ncols=1)

    if colors is None:
        
        council_df = pd.read_csv(path + 'council_info.csv')
        colors = council_df['Color']
        party_color_map = {'UDC': 'royalblue',
                           'PSS': 'r',
                           'PDC': 'orange',
                           'pvl': 'g',
                           'PLR': 'cyan',
                           'PES': 'forestgreen',
                           'PBD': 'yellow'}
        
        patch_list = []
        for key in party_color_map:
            data_key = mpl.patches.Patch(color=party_color_map[key], label=key)
            patch_list.append(data_key)

        ax.legend(handles=patch_list,
                  frameon=False,
                  facecolor='inherit',
                  fontsize=fontsize)
        
    ax.scatter(coords[:,0], coords[:,1], c=colors, s=100, alpha=0.8, 
               edgecolors='none')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    
    if save_to_disk:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        save_path = save_dir + now + ' ' + file_name + '.pdf'
        plt.savefig(save_path, dpi=dpi, transparent=True)
        
    return fig, ax
