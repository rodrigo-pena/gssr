#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Graphs and Signals module

"""


import numpy as np
import pandas as pd
import networkx as nx

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
        labels = np.zeros((n_vertices))
        cnt = n_vert_per_comm[0]
        comm_label = 0
        for number in n_vert_per_comm[1:]:
            comm_label += 1
            labels[cnt:cnt+number] = comm_label
            cnt += number
        labels[cnt:] = comm_label # deal with the rest
    labels = labels.astype(int)
    
    # Default is above the the connectivity threshold for 2-SSBM
    p = 2 * np.log(n_vertices)/n_vertices if intra_comm_prob is None else intra_comm_prob
    q = 1 * np.log(n_vertices)/n_vertices if inter_comm_prob is None else inter_comm_prob
    
    # Call SBM object from `pygsp`
    graph = pygsp.graphs.StochasticBlockModel(N=n_vertices, k=n_communities, z=labels, 
                                              M=comm_prob_mat, p=p, q=q, seed=seed)
    graph.set_coordinates(kind='community2D')
    
    # Assemble the indicator vectors from the community labels
    indicator_vectors = np.zeros((n_communities, n_vertices))
    for k in np.arange(n_communities):
        indicator_vectors[k, :] = (z == k).astype(float)
    
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


def swiss_nacional_council(path='data/swiss-national-council-50/'):
    r"""
    Graph and signals for the Swiss National Council data.
    
    Parameters
    ----------
    path : str
        The path to the directory containing the files `council_info.csv` and `adjacency.csv`.
    
    Returns
    -------
    :class:`pygsp.graphs.Graph`
        The graph object.
    ndarray
        A p-by-n matrix containing the indicator vectors of each of the p parties.
        
    """
    
    council_df = pd.read_csv(path + 'council_info.csv')
    adjacency = pd.read_csv(path + 'adjacency.csv')
    
    graph = pygsp.graphs.Graph(adjacency=adjacency)
    graph.compute_laplacian(lap_type='normalized')
    graph.compute_fourier_basis()
    graph.set_coordinates(kind='laplacian_eigenmap2D')
    
    parties = ['UDC', 'PSS', 'PDC', 'pvl', 'PLR', 'PES', 'PBD']
    n_parties = len(parties)
    indicator_vectors = np.zeros((n_parties, graph.n_vertices))
    for i in np.arange(n_parties):
        indicator_vectors[i, :] = np.asarray(council_df['PartyAbbreviation']==parties[i]).astype(float)
    
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
    indicator_vectors = np.zeros((n_communities, graph.n_vertices))
    for i in np.arange(n_communities):
        indicator_vectors[i, :] = (labels == i).astype(float)
    
    return graph, indicator_vectors