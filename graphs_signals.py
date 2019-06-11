#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Graphs and Signals module

"""


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import os
import utils
import pygsp


def sbm(n_vertices, n_communities, n_vert_per_comm=None, comm_prob_mat=None, intra_comm_prob=None,
        inter_comm_prob=None, seed=None):
    r"""
    Draw a graph from the Stochastic Block Model (SBM).
    
    Parameters
    ----------
    n_vertices : int
        Number of vertices in the graph.
    n_communities : int
        Number of communities (clusters, classes) in the graph.
    n_vert_per_comm : array_like, optional
        A list of size `n_communities` indicating how many vertices are to be but in each community.
        (default is equal-sized communities)
    comm_prob_mat : ndarray, optional
        A `n_communities`-by-`n_communities` matrix containing the connection probabilities among the 
        communities. The diagonal entries correspond to intra-community probabilities, whereas 
        off-diagonal entries correspond to inter-community probabilities.
        (default is uniform and given by `intra_comm_prob` and `inter_comm_prob`)
    intra_comm_prob : array_like
        A list containing the intra-community connectivity probabilities. If scalar, then uniform 
        connection probabilities of this value are assumed.
        (default is `2 * np.log(n_vertices)/n_vertices`)
    inter_comm_prob : array_like
        A list containing the inter-community connectivity probabilities. If scalar, then uniform 
        connection probabilities of this value are assumed.
    seed : float
        A seed for the random number generators, to generate reproducible graphs.
        (default is None)
    
    Returns
    -------
    :class:`pygsp.graphs.Graph`
        The graph object.
    ndarray
        A k-by-n matrix containing the indicator vectors of each of the k communities.
    
    """
    
    # Construct a vector assigning a class label to each vertex
    # (needed by :class:`pygsp.graphs.StochasticBlockModel`)
    if n_vert_per_comm is None:
        rest = np.mod(n_vertices, n_communities).astype(int)
        labels = np.repeat(np.arange(0, n_communities), n_vertices/n_communities) # Balanced classes
        labels = np.append(labels, (n_communities-1) * np.ones(rest,)) # deal with the rest
    else:
        if n_communities - np.asarray(n_vert_per_comm).shape[0] > 0:
            n_vert_per_comm = np.append(n_vert_per_comm, n_vertices - np.sum(n_vert_per_comm))
            print(n_vert_per_comm)
        labels = np.zeros((n_vertices,))
        count = n_vert_per_comm[0]
        comm_label = 0
        for number in n_vert_per_comm[1:]:
            comm_label += 1
            labels[count:count+number] = comm_label
            count += number
        labels[count:] = comm_label # deal with the rest
    labels = labels.astype(int)
    
    # Default is above the the connectivity threshold for 2-SSBM
    p = 2 * np.log(n_vertices)/n_vertices if intra_comm_prob is None else intra_comm_prob
    q = 1 * np.log(n_vertices)/n_vertices if inter_comm_prob is None else inter_comm_prob
    
    # Call SBM object from `pygsp`
    graph = pygsp.graphs.StochasticBlockModel(N=n_vertices, k=n_communities, z=labels, 
                                              M=comm_prob_mat, p=p, q=q, seed=seed)
    graph.set_coordinates(kind='community2D')
    graph.info['n_communities'] = n_communities
    
    # Assemble the indicator vectors from the community labels
    indicator_vectors = np.zeros((n_communities, n_vertices))
    for k in np.arange(n_communities):
        indicator_vectors[k, :] = (labels == k).astype(float)
    
    return graph, indicator_vectors


def ssbm(n_vertices, n_communities=2, a=2., b=1., seed=None):
    r"""
    Draw a graph from the Symetric Stochastic Block Model (SBM).
    
    Parameters
    ----------
    n_vertices : int
        Number of vertices in the graph. The true number of vertices may be reduced to make 
        sure that each community has the exact same number of vertices.
    n_communities : int, optional
        Number of communities (clusters, classes) in the graph.
        (default is `2`)
    a : float
        A scalar multiplying `np.log(n_vertices)/n_vertices` to yield the intra-community 
        connection probabilities.
        (default is `2.`)
    b : float
        A scalar multiplying `np.log(n_vertices)/n_vertices` to yield the inter-community 
        connection probabilities.
        (default is `2.`)
    seed : float
        A seed for the random number generators, to generate reproducible graphs.
        (default is None)
    
    Returns
    -------
    :class:`pygsp.graphs.Graph`
        The graph object.
    ndarray
        A k-by-n matrix containing the indicator vectors of each of the k communities.
    
    """
    
    # Ensure an identical number of vertices per community
    n_vertices -= np.mod(n_vertices, n_communities)
    
    # Intra- and inter-community connection probabilities
    p = a * np.log(n_vertices)/n_vertices
    q = b * np.log(n_vertices)/n_vertices
    
    return sbm(n_vertices, n_communities, intra_comm_prob=p, inter_comm_prob=q, seed=seed)


def swiss_national_council(path='data/swiss-national-council/', 
                           councillors_fn='councillors.csv',
                           affairs_fn='affairs.csv', 
                           voting_matrix_fn='voting_matrix.csv',
                           **kwargs):
    r"""
    Graph and signals for the Swiss National Council data.
    
    Parameters
    ----------
    path : str, optional
        The path to the directory containing the councillor, affairs, and voting matrix `.csv` files. 
        (default is 'data/swiss-national-council/')
    councillors_fn : str, optional
        Name of the councillor info file. (default is 'councillors.csv')
    affairs_fn : str, optional
        Name of the voting affairs info file (default is 'affairs.csv')
    voting_matrix_fn : str, optional
        Name of the voting matrix file (default is 'voting_matrix.csv')
    kwargs : dict
        Extra parameters passed to :class:`pygsp.graph.NNGraph`. 
    
    Returns
    -------
    :class:`pygsp.graphs.Graph`
        The graph object.
    ndarray
        A p-by-n matrix containing the indicator vectors of each of the p parties.
        
    """
    
    # Read .csv files
    councillors = pd.read_csv(os.path.join(path, councillors_fn))
    affairs = pd.read_csv(os.path.join(path, affairs_fn))
    voting_matrix = pd.read_csv(os.path.join(path, voting_matrix_fn)).values
    
    # Get parties and member counts
    from collections import Counter
    party_count = Counter(councillors['PartyAbbreviation'])
    parties = np.asarray(list(party_count.keys())).astype(str)
    members_per_party = np.asarray(list(party_count.values())).astype(int)
    n_communities = len(parties)
    n_vertices = np.sum(members_per_party)
    
    # Sort councillors from largest to smallest party, and create label vectors
    sort_idx = np.argsort(members_per_party)[-1::-1]
    members_per_party = members_per_party[sort_idx]
    parties = parties[sort_idx]
    
    councillors_ordered = pd.DataFrame(columns=councillors.columns)
    
    for i in np.arange(n_communities):
        party_mask = (councillors['PartyAbbreviation'] == parties[i])
        councillors_ordered = councillors_ordered.append(councillors[party_mask])
    
    voting_matrix = voting_matrix[councillors_ordered.index.values, :]
    councillors_ordered = councillors_ordered.reset_index(drop=True)
    
    # Create party label vectors
    labels = np.nan * np.ones((n_vertices,))
    indicator_vectors = np.zeros((n_communities, n_vertices))
    
    for i in np.arange(n_communities):
        party_mask = (councillors_ordered['PartyAbbreviation'] == parties[i])
        indicator_vectors[i, :] = np.asarray(party_mask).astype(float)
        labels[party_mask] = i
        
    labels = labels.astype(int)
    
    # Create Nearest-Neighbors graph
    graph = pygsp.graphs.NNGraph(voting_matrix, **kwargs)
    
    graph.info = {
        'node_com': labels, 
        'comm_sizes': members_per_party, 
        'world_rad': np.sqrt(graph.n_vertices),
        'parties': parties,
        'n_communities': n_communities,
        'councillors': councillors_ordered,
        'affairs': affairs
    }
    
    graph.compute_fourier_basis()
    graph.set_coordinates(kind='laplacian_eigenmap2D')
    
    return graph, indicator_vectors


def email_eu_core(path='data/email-EU-core/'):
    r"""
    Graph and signals for the email-Eu-core data.
    
    Parameters
    ----------
    path : str
        The path to the directory containing the files `email-Eu-core.txt` and 
        `email-Eu-core-department-labels.txt`.
    
    Returns
    -------
    :class:`pygsp.graphs.Graph`
        The graph object.
    ndarray
        A p-by-n matrix containing the indicator vectors of each of the p parties.
        
    Notes
    -----
    Data source: http://snap.stanford.edu/data/email-Eu-core.html
        
    """
    
    edgelist = np.loadtxt(path + 'email-Eu-core.txt')
    labels = np.loadtxt(path + 'email-Eu-core-department-labels.txt').astype(int)[:,1]

    graph_nx = nx.from_edgelist(edgelist)
    graph_nx.remove_edges_from(graph_nx.selfloop_edges())

    graph = pygsp.graphs.Graph(adjacency=nx.adj_matrix(graph_nx))
    graph.info = {'node_com': labels, 
                  'comm_sizes': np.bincount(labels), 
                  'world_rad': np.sqrt(graph.n_vertices)}
    graph.set_coordinates(kind='community2D')
    
    n_communities = len(graph.info['comm_sizes'])
    graph.info['n_communities'] = n_communities
    
    indicator_vectors = np.zeros((n_communities, graph.n_vertices))
    
    for i in np.arange(n_communities):
        indicator_vectors[i, :] = (labels == i).astype(float)
    
    return graph, indicator_vectors


def bsds300(img_id, path='data/BSDS300/', seg_subset='color', subsample_factor=12, 
            graph_type='grid', patch_shape=(7,7), **kwargs):
    r"""
    Graph and signals for the BSDS300 data.
    
    Parameters
    ----------
    img_id : str
        ID of the image. See `iids_train.txt` and `iids_test.txt` files.
    path : str, optional
        The path to the directory containing the BSDS300 dataset. 
        (default is 'data/BSDS300/')
    seg_subset : str, optional
        Segmentation data subset to use. Options are 'color' or 'gray'.
        (default is 'color')
    subsample_factor : int
        Factor by which to subsample the images for a better run time. 
        (default is 12)
    graph_type : str
        Graph structure type. Options are 'grid', 'patches', and 'grid_and_patches'.
        The first option produces a graph whose egde structure reflects exactly the
        pixel grid of the image. The second option constructs k-Nearest Neighbors
        graph by connecting pixels that live in similar patches. The third option is
        a combination of the other two.
        (default is 'grid')
    patch_shape : tuple
        Shape of the pixel patch used for computing pixel similarity. Only used if 
        `graph_type` is 'patches' or 'grid_and_patches'.
        (default is `(7,7)`)
    kwargs : dict
        Extra parameters passed to :class:`pygsp.graph.NNGraph`. 
    
    Returns
    -------
    :class:`pygsp.graphs.Graph`
        The graph object.
    ndarray
        A k-by-n matrix containing the indicator vectors of each of the 
        k segmentation classes for the image.
        
    Notes
    -----
    If you pick 'patches' or 'grid_and_patches' as `graph_type`, consider setting
    `use_flann=True` in `kwargs` for faster computation.
    
    Data source: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
        
    """
    
    from glob import glob, iglob
    
    # Image #
    img_file = glob(os.path.join(path, 'images/**/' + img_id +'.jpg'), recursive=True)[0]
    img = plt.imread(img_file)
        
    # Segmentation data #
    
    if seg_subset == 'color':
        seg_path = path + 'human/color/'
    elif seg_subset == 'gray':
        seg_path = path + 'human/gray/'
    else:
        raise ValueError("Valid options for seg_subset are 'color' or 'gray'.")
    
    # TODO: is there a better choice of file than the first one in the list?
    seg_file = glob(os.path.join(seg_path, '**/' + img_id +'.seg'), recursive=True)[0]
    
    # Get header and data, following instructions from `seg-format.txt`
    seg_header = pd.read_csv(seg_file, names=[0, 1], sep=' ', nrows=11)
    seg_data = pd.read_csv(seg_file, skiprows=11, names=[0, 1, 2, 3], sep=' ').values
    
    # Build segmentation mask
    seg_mask = np.zeros((int(seg_header[0]['height']), int(seg_header[0]['width'])))
    count = 0
    
    for i in seg_data[:, 1]:
        seg_mask[i, seg_data[count, 2]:seg_data[count, 3]] = seg_data[count, 0]
        count += 1
    
    # Subsample image and segmentation mask by the same factor
    img = img[::subsample_factor, ::subsample_factor, :]
    h, w, c = img.shape
    seg_mask = seg_mask[::subsample_factor, ::subsample_factor]
    
    # Build label vector
    labels = seg_mask.ravel().astype(int)
    
    # Graph #
    
    graph = pygsp.graphs.Grid2d(N1=h, N2=w)
    coords = graph.coords
    
    if graph_type == 'grid':
        pass
    
    elif graph_type == 'patches':        
        graph = pygsp.graphs.ImgPatches(img, patch_shape=patch_shape, **kwargs)
       
    elif graph_type == 'grid_and_patches':
        graph = pygsp.graphs.Grid2dImgPatches(img, patch_shape=patch_shape, **kwargs)
        
    else:
        raise ValueError("Valid options for graph_type are 'grid', 'patch', or 'grid_and_patch'.")
    
    graph.img = img
    graph.coords = coords
    
    graph.info = {
        'node_com': labels,
        'comm_sizes' : np.bincount(labels),
        'n_communities' : len(np.bincount(labels)),
        'world_rad' : np.sqrt(graph.n_vertices),
        'img_id' : img_id,
        'img_subset' : utils.get_bsds300_subset(img_id, path),
        'seg_subset' : seg_subset
    }
    
    # Indicator vectors #
    
    indicator_vectors = np.zeros((graph.info['n_communities'], graph.n_vertices))
    
    for i in np.arange(graph.info['n_communities']):
        indicator_vectors[i, :] = (labels == i).astype(float) 
    
    return graph, indicator_vectors
