#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Sampling module

"""

import cvxpy

import numpy as np

from scipy import sparse


## MAIN FUNCTIONS ##

def sample_coordinates(n_coordinates, n_samples, probs=None, replace=False):
    r"""
    Sample a given number of coordinates from a specified probability mass function

    Parameters
    ----------
    n_coordinates : int
        Number of coordinates from which to sample
    n_samples : int
        Number of samples to take from the set `{1, 2, ..., n_coordinates}`.
    probs : array, optional
        A vector with the probability weights for each coordinate.
        (the default is None, which results in uniform sampling)
    replace: bool, optional
        Sample with replacement. (default is False)

    Returns
    -------
    array
        The indices of the sampled coordinates
        
    """
    
    n_coordinates = int(n_coordinates)
    n_samples = int(n_samples)
    
    if probs is not None:
        probs = np.true_divide(probs, np.sum(probs))
    
    return np.random.choice(n_coordinates, n_samples, replace=replace, p=probs)


def sample_coordinates_bernoulli(n_coordinates, probs=None):
    r"""
    Sample coordinates using Bernoulli selectors

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
    Uniform sampling over the vertices

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
    Sample vertices proportionally to their degree

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
    Sample vertices proportionally to the inverse of their degree

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
    Sampling proportional to connection with members of other communities

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


def naive_coherence_diff_std_basis(graph, n_samples, replace=False):
    r"""
    Sample vertices proportionally to the inverse of a naive measure of coherence.

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
        
    """
    
    graph.compute_differential_operator() # Make sure the graph has D as an attribute
    D = graph.D.toarray().T.copy()        # The incidence matrix in graph.D is the 
                                          # transpose of the gradient matrix
    D_plus = np.linalg.pinv(D)            # Pseudoinverse of the gradient matrix
    
    # Sampling probability weights: product of the row-wise infinity-norm of D_plus_T times
    # the row-wise l1-norm of D
    probs = np.max(np.abs(D_plus.T), axis=0) * np.sum(np.abs(D), axis=0)
    
    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=probs,
                              replace=replace)


def coherence_cut_diff_std_basis(graph, n_samples, gt_signal, replace=False):
    r"""
    Sample vertices proportionally to the inverse of a measure of coherence over the true cut.

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
        
    """
    
    graph.compute_differential_operator() # Make sure the graph has D as an attribute
    D = graph.D.toarray().T.copy()        # The incidence matrix in graph.D is the 
                                          # transpose of the gradient matrix
    D_plus = np.linalg.pinv(D)            # Pseudoinverse of the gradient matrix
    
    # Orthogonal projection matrix onto the support of the cut 
    P_S = np.diag((np.abs(D @ gt_signal) > 1e-6).astype(float))
    
    # Sampling probability weights: product of the row-wise infinity-norm of D_plus_T times
    # the row-wise l1-norm of D
    probs = np.max(np.abs(D_plus.T), axis=0) * np.sum(np.abs(P_S @ D), axis=0)
    
    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=probs,
                              replace=replace)


