#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plotting module

"""


import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


def pt_grid(grid, ax=None, add_cbar=True, vmin=None, vmax=None):
    r"""
    Plot in the style of a phase transition grid.
    
    Parameters
    ----------
    grid : array_like
        The pixel grid to plot.
    ax : :class:`matplotlib.axes.Axes`, optional
        Container for figure elements.
    add_cbar : bool, optional
        Include colorbar on plot. (default is True)
    vmin : float, optional
        Minimum value in `matplotlib.pyplot.imshow`.
    vmax : float, optional
        Maximum value in `matplotlib.pyplot.imshow`.
    
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
    fig = None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, ncols=1) 
    else:
        fig = plt.gcf()
        
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
        
    return fig, ax    
    

def snc_with_party_labels(graph, ax=None):
    r"""
    Scatter plot of the Swiss National Council members.

    Parameters
    ---------
    graph : :class:`pygsp.graphs.Graph`
        The graph object for the Swiss National Council members.
    ax : :class:`matplotlib.axes.Axes`, optional
        Container for figure elements.
    kwargs: dict
        Other parameters for :func:`pygsp.graphs.Graph.plot()`.
        
    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Top level container for the plot element.
    
    ax : :class:`matplotlib.axes.Axes`
        Container for figure elements.
        
    """
    
    # Global settings
    fontsize = 14
    dpi = 300
    
    # Color map
    # Source: en.wikipedia.org/wiki/List_of_political_parties_in_Switzerland
    party_color_map = {'UDC': '#13923E',
                       'PSS': '#DB182A',
                       'FDP': '#0E3D8F',
                       'PDC': '#E96807',
                       'PBD': '#FED809',
                       'PES': '#73A812',
                       'PVL': '#97C834',
                       'PEV': '#FDD80B',
                       'Lega': '#527FE8',
                       'MCG': '#FDE609',
                       'PST': '#E02416',
                       'CSPO': '#AF1E28',
                       'CSP': '#168397',
                       'UDF': '#B80072',
                       'AL': '#820013'
                       } 
    
    # Legend patches
    patch_list = []
    
    for key in graph.info['parties']:
        
        try:
            # Set alpha=0.5 to match pygsp plot
            c = mpl.colors.to_rgba(party_color_map[key], alpha=0.5)
            patch_list.append(mpl.patches.Patch(color=c, label=key))
            
        except KeyError: # key not in party_color_map
            pass
            
    patch_list.append(mpl.patches.Patch(color='k', label='others'))
    
    # Plot 
    fig = None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), ncols=1)
    else:
        fig = plt.gcf()

    colors = graph.info['councillors']['PartyColor'].values
    graph.plot(colors, ax=ax, **kwargs)
    
    ax.set_title('')
    ax.set_axis_off()
    ax.legend(handles=patch_list, 
              frameon=False, 
              facecolor='inherit', 
              fontsize=fontsize,
              bbox_to_anchor=(1.05, 1),
              loc=2, 
              borderaxespad=0.)
    
    return fig, ax
