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


def plot_graph(graph, *args, ax=None, **kwargs):
    
    fig = None
    
    if ax is None:
        fig, ax = plt.subplots(ncols=1) 
    else:
        fig = plt.gcf()
    
    graph.plot(*args, ax=ax, **kwargs)
    
    ax.set(title='')
    
    ax.grid(False)
    
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False, 
                   top=False, 
                   left=False, 
                   right=False,
                   labelbottom=False, 
                   labeltop=False, 
                   labelleft=False, 
                   labelright=False)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return fig, ax
 

def make_snc_legend(ax):
    r"""
    Party color legend for the Swiss National Council.

    Parameters
    ---------
    ax : :class:`matplotlib.axes.Axes`, optional
        Container for figure elements.
        
    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Container for figure elements.
        
    """
    
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
    
    for key in party_color_map:
        color = mpl.colors.to_rgba(party_color_map[key], alpha=1.0)
        patch_list.append(mpl.patches.Patch(color=color, label=key))
        
    patch_list.append(mpl.patches.Patch(color='k', label='others'))
    
    ax.legend(handles=patch_list, 
              frameon=False, 
              facecolor='inherit', 
              bbox_to_anchor=(1.05, 1),
              loc='upper left', 
              borderaxespad=0.)
    
    return ax
