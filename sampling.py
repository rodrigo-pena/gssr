#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Sampling module

"""


import numpy as np

from scipy import sparse


## MAIN FUNCTIONS ##

def sample_coordinates(n_coordinates, n_samples, probs=None, replace=False):
    r"""
    Sample a fixed number of coordinates independently at random.

    Parameters
    ----------
    n_coordinates : int
        Number of coordinates from which to sample.
    n_samples : int
        Number of samples to take from the set `{0, 1, 2, ..., n_coordinates - 1}`.
    probs : array, optional
        The probability weights for each coordinate.
        (the default is None, which results in uniform sampling)
    replace: bool, optional
        Sample with replacement. (default is False)

    Returns
    -------
    array
        The indices of the sampled coordinates from the set 
        `{0, 1, 2, ..., n_coordinates - 1}`.
        
    """
    
    n_coordinates = int(n_coordinates)
    n_samples = int(n_samples)
    
    if probs is not None:
        probs = np.true_divide(probs, np.sum(probs))
    
    return np.random.choice(n_coordinates, n_samples, replace=replace, p=probs)


def sample_coordinates_bernoulli(n_coordinates, probs=None):
    r"""
    Sample coordinates using Bernoulli selectors.

    Parameters
    ----------
    n_coordinates : int
        Number of coordinates from which to sample
    probs : array, optional
        A vector with the success probabilities of each coordinate selector.
        (the default is None, which results in a fair coin toss for each coordinate)

    Returns
    -------
    array
        The indices of the sampled coordinates
        
    Notes
    -----
    This sampling is always without replacement, but the number of samples is a random
    variable.
        
    """
    
    n_coordinates = int(n_coordinates)
    sample_idx = []
    probs = np.ones((n_coordinates,))/2. if probs is None else probs

    for i in np.arange(n_coordinates):
        if np.random.binomial(1, probs[i]) == 1:
            sample_idx.append(i)

    return np.array(sample_idx)


## SPECIAL CASES ##

def uniform_vertex(graph, n_samples, replace=False):
    r"""
    Sample vertices uniformly at random.

    Parameters
    ----------
    graph: :class:`pygsp.graphs.Graph`
        A graph object. 
    n_samples: int
        Number of measurements to take
    replace: bool, optional
        Sample with replacement. (default is True)

    Returns
    -------
    (n_samples, ) array
        Indices of the sampled vertices
        
    """

    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=None,
                              replace=replace)


def degree_vertex(graph, n_samples, replace=False):
    r"""
    Sample vertices proportionally to their degree.

    Parameters
    ----------
    graph: :class:`pygsp.graphs.Graph`
        A graph object. 
    n_samples: int
        Number of measurements to take
    replace: bool, optional
        Sample with replacement. (default is True)

    Returns
    -------
    (n_samples, ) array
        Indices of the sampled vertices
        
    """
    
    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=graph.dw,
                              replace=replace)


def inv_degree_vertex(graph, n_samples, replace=False):
    r"""
    Sample vertices proportionally to the inverse of their degree.

    Parameters
    ----------
    graph: :class:`pygsp.graphs.Graph`
        A graph object. 
    n_samples: int
        Number of measurements to take
    replace: bool, optional
        Sample with replacement. (default is True)

    Returns
    -------
    (n_samples, ) array
        Indices of the sampled vertices
        
    """

    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=1./graph.dw,
                              replace=replace)


def inter_comm_degree_vertex(graph, n_samples, gt_signal, replace=False):
    r"""
    Sample vertices proportionally to their connection weight with members in other communities

    Parameters
    ----------
    graph: :class:`pygsp.graphs.Graph`
        A graph object. 
    n_samples: int
        Number of measurements to take
    gt_signal : array
        The ground truth signal to be sampled.
    replace: bool, optional
        Sample with replacement. (default is True)

    Returns
    -------
    (n_samples, ) array
        Indices of the sampled vertices
        
    Notes
    -----
    A community is defined as a level-set of the ground-truth signal. For example, if `gt_signal`
    is {0,1}-valued, one community will contain all the vertices `i` for which `gt_signal[i]=0`,
    while the other community will contain all the vertices `j` for which `gt_signal[i]=1`.
        
    """

    # Copy the (weigthed) adjacency matrix
    adjacency = sparse.lil_matrix(graph.W.copy())

    # Make sure the labels vector is a numpy array
    labels = np.asarray(gt_signal)

    for vertex in np.arange(len(labels)):
        # Disconnect vertices that belong in the same community
        adjacency[vertex, labels == labels[vertex]] = 0

    inter_comm_degree = np.array(adjacency.sum(axis=1)).flatten()

    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=inter_comm_degree,
                              replace=replace)


def naive_tv_coherence(graph, n_samples, replace=False):
    r"""
    Sample proportionally to a naive coherence measure from the TV interpolation problem.

    Parameters
    ----------
    graph: :class:`pygsp.graphs.Graph`
        A graph object. 
    n_samples: int
        Number of measurements to take.
    replace: bool, optional
        Sample with replacement. (default is True)

    Returns
    -------
    (n_samples, ) array
        Indices of the sampled vertices.
        
    Notes
    -----
    Denote by :math:`D` the graph gradient matrix, and let :math:`\{e_i\}_i` be the
    standard basis in :math:`\mathbb{R}^n`. The vertex sampling probabilities are set as 
    :math:`\pi_i \propto \|(D^+)^\top e_i\|_\infty \cdot \|D e_i\|_1`,
    the product of the column-wise infinity-norm of :math:`(D^+)^\top` with
    the column-wise l1-norm of :math:`D`.
        
    """
    
    # Make sure the graph has D as an attribute
    graph.compute_differential_operator() 
    
    # The incidence matrix in graph.D is the transpose of the gradient matrix
    D = graph.D.toarray().T.copy()    
    
    # Get the Moore-Penrose pseudo-inverse of the gradient matrix
    D_plus = np.linalg.pinv(D)            
    
    # Set the sampling probability weights
    probs = np.max(np.abs(D_plus.T), axis=0) * np.sum(np.abs(D), axis=0)
    
    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=probs,
                              replace=replace)


def jump_set_tv_coherence(graph, n_samples, gt_signal, replace=False):
    r"""
    Sample proportionally to a jump-set-restricted coherence measure from the TV interpolation problem.
    
    Parameters
    ----------
    graph: :class:`pygsp.graphs.Graph`
        A graph object. 
    n_samples: int
        Number of measurements to take.
    gt_signal : array
        The ground truth signal to be sampled.
    replace: bool, optional
        Sample with replacement. (default is True)

    Returns
    -------
    (n_samples, ) array
        Indices of the sampled vertices.
        
    Notes
    -----
    Denote by :math:`D \in \mathbb{R}^{N \times n}` the graph gradient matrix, and by
    :math:`x` the ground-truth signal to be sampled. Let :math:`\{e_i\}_i` be the 
    standard basis in :math:`\mathbb{R}^n` and :math:`P_S` be the orthogonal projection
    of vectors in :math:`\mathbb{R}^N` onto the support :math:`S`of :math:`Dx`. The 
    vertex sampling probabilities are set as 
    :math:`\pi_i \propto \|(D^+)^\top e_i\|_\infty \cdot \|P_S D e_i\|_1`,
    the product of the column-wise infinity-norm of :math:`(D^+)^\top` with
    the column-wise l1-norm of :math:`P_S D`.
    
    """
    
    # Make sure the graph has D as an attribute
    graph.compute_differential_operator() 
    
    # The incidence matrix in graph.D is the transpose of the gradient matrix
    D = graph.D.toarray().T.copy()    
    
    # Get the Moore-Penrose pseudo-inverse of the gradient matrix
    D_plus = np.linalg.pinv(D)   
    
    # Get the orthogonal projection matrix onto the support of the jump-set 
    P_S = np.diag((np.abs(D @ gt_signal) > 1e-6).astype(float))
    
    # Set the sampling probability weights
    probs = np.max(np.abs(D_plus.T), axis=0) * np.sum(np.abs(P_S @ D), axis=0)
    
    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=probs,
                              replace=replace)


