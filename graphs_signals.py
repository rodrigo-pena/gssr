#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Graphs and Signals module

"""

from collections import Counter

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
        
    Examples
    --------
    >>> import graphs_signals as gs
    >>> graph, indicator_vectors = gs.ssbm(n_vertices=100, n_communities=2, n_vert_per_comm=[20, 80])
    >>> graph.plot(indicator_vectors[0,:])
    
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
    
    Examples
    --------
    >>> import graphs_signals as gs
    >>> graph, indicator_vectors = gs.ssbm(n_vertices=100)
    >>> graph.plot(indicator_vectors[0,:])
    
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
        
    Examples
    --------
    >>> import graphs_signals as gs
    >>> nn_params = {'NNtype': 'knn',
                     'use_flann': True,
                     'center': False,
                     'rescale': True,
                     'k': 25,
                     'dist_type': 'euclidean'}
    >>> graph, indicator_vectors = gs.swiss_national_council(**nn_params)
    >>> graph.plot(indicator_vectors[0,:])
        
    """
    
    # Read .csv files
    councillors = pd.read_csv(os.path.join(path, councillors_fn))
    affairs = pd.read_csv(os.path.join(path, affairs_fn))
    voting_matrix = pd.read_csv(os.path.join(path, voting_matrix_fn)).values
    
    # Get parties and member counts
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
    
    graph.coords = utils.get_parliament_coordinates(n_councillors=graph.n_vertices)
    
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
    
    Examples
    --------
    >>> import graphs_signals as gs
    >>> graph, indicator_vectors = gs.email_eu_core()
    >>> graph.plot(indicator_vectors[0,:])
        
    """
    
    edgelist = np.loadtxt(path + 'email-Eu-core.txt')
    labels = np.loadtxt(path + 'email-Eu-core-department-labels.txt').astype(int)[:,1]
    
    # Build graph
    graph_nx = nx.from_edgelist(edgelist)
    graph_nx.remove_edges_from(graph_nx.selfloop_edges())
    
    adjacency = nx.adj_matrix(graph_nx)

    graph = pygsp.graphs.Graph(adjacency=adjacency)
    graph.info = {'node_com': labels, 
                  'comm_sizes': np.bincount(labels), 
                  'world_rad': 3 * np.sqrt(graph.n_vertices)}
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
    
    Examples
    --------
    >>> import graphs_signals as gs
    >>> graph, indicator_vectors = gs.bsds300('159029', graph_type='grid_and_patches', k=3, use_flann=True)
    >>> graph.plot(graph.info['node_com'])
        
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


def high_school_social_network(path='data/high-school/', kind='contact'):
    r"""
    Graph and signals for the high school contact and friendship data.
    
    Parameters
    ----------
    path : str, optional
        The path to the directory containing the high-school dataset. 
        (default is 'data/high-school/')
    kind : str, optional
        {'contact', 'friendship', 'facebook', 'merge-all'} Type of network to load.
        (default is 'contact')
    kwargs : dict
        Extra parameters.
    
    Returns
    -------
    :class:`pygsp.graphs.Graph`
        The graph object.
    ndarray
        A k-by-n matrix containing the indicator vectors of each of the 
        k classes in the school.
        
    Notes
    -----
    
    
    Examples
    --------
    >>> import graphs_signals as gs
    >>> graph, indicator_vectors = gs.high_school_social_network(kind='contact')
    >>> graph.plot(indicator_vectors[0,:])
        
    """
    
    # Read metadata
    metadata_fn = 'metadata_2013.txt'
    meta_df = pd.read_csv(path + metadata_fn, delimiter='\t', engine='python', header=None, 
                 names=['id', 'class', 'gender'], 
                 dtype={'id': np.int32, 'class': str, 'gender': str})
    meta_df = meta_df.sort_values(by=['id']).reset_index(drop = True)
    
    # Get class and people counts
    class_count = Counter(meta_df['class'])
    classes = np.asarray(list(class_count.keys())).astype(str)
    people_per_class = np.asarray(list(class_count.values())).astype(int)
    n_communities = len(classes)
    n_vertices = np.sum(people_per_class)
    
    # Sort classes from largest to smallest
    sort_idx = np.argsort(people_per_class)[-1::-1]
    people_per_class = people_per_class[sort_idx]
    classes = classes[sort_idx]
    
    meta_df_ordered = pd.DataFrame(columns=meta_df.columns)
    
    for i in np.arange(n_communities):
        class_mask = (meta_df['class'] == classes[i])
        meta_df_ordered = meta_df_ordered.append(meta_df[class_mask])
    
    people_sort_idx = meta_df_ordered.index.values
    meta_df_ordered = meta_df_ordered.reset_index(drop=True)
    
    # Create class label vectors
    labels = np.nan * np.ones((n_vertices,))
    indicator_vectors = np.zeros((n_communities, n_vertices))
    
    for i in np.arange(n_communities):
        class_mask = (meta_df_ordered['class'] == classes[i])
        indicator_vectors[i, :] = np.asarray(class_mask).astype(float)
        labels[class_mask] = i
        
    labels = labels.astype(int)
    
    # Read graph edgelist
    if kind == 'contact':
        fn = 'Contact-diaries-network_data_2013.csv'
        
    elif kind == 'friendship':
        fn = 'Friendship-network_data_2013.csv'
        
    elif kind == 'facebook':
        fn = 'Facebook-known-pairs_data_2013.csv'
        
    elif kind == 'merge-all': # Recursion time!
        
        graph_contact, _ = high_school_social_network(path=path, kind='contact')
        graph_friend, _ = high_school_social_network(path=path, kind='friendship')
        graph_face, indicator_vectors = high_school_social_network(path=path, kind='facebook')
        
        adjacency = graph_contact.W + graph_friend.W + graph_face.W
        
        graph = pygsp.graphs.Graph(adjacency=adjacency)
        graph.info = graph_face.info
        graph.coords = graph_face.coords
        
        return graph, indicator_vectors
        
    else:
        raise ValueError("Valid options for kind are 'contact', 'friendship' or 'facebook'.")
    
    edgelist_df = pd.read_csv(path + fn, 
                              delimiter=' ', 
                              engine='python', 
                              header=None, 
                              names=['source', 'target', 'weight'])
    
    if np.isnan(edgelist_df['weight']).all(): # True for the friendship network
        edgelist_df['weight'] = 1.0
    
    # Undirected networkx graph from pandas edgelist
    graph_nx = nx.from_pandas_edgelist(edgelist_df, 
                                       source='source', 
                                       target='target', 
                                       edge_attr='weight')
    
    
    # Graph
    ids = np.array(meta_df_ordered['id'].values) # length = n_vertices
    ids_with_connections = np.sort(list(graph_nx.nodes))
    id_mask = np.isin(ids, ids_with_connections)
    
    adjacency = np.zeros((n_vertices, n_vertices))
    adjacency[np.ix_(id_mask, id_mask)] = nx.adj_matrix(graph_nx).toarray()
    adjacency = adjacency[np.ix_(people_sort_idx, people_sort_idx)]
    
    graph = pygsp.graphs.Graph(adjacency=adjacency)
    
    graph.info = {
        'node_com': labels,
        'comm_sizes' : np.bincount(labels),
        'n_communities' : len(np.bincount(labels)),
        'world_rad' : 1.5 * np.sqrt(graph.n_vertices),
        'kind' : kind
    }
    graph.set_coordinates(kind='community2D')
    
    return graph, indicator_vectors