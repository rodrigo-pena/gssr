#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Sampling module

"""


import numpy as np

from scipy import sparse


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
    replace : bool, optional
        Whether to sample with replacement
        (the default is False, which results in sampling without replacement).

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


def uniform_vertex(graph, n_samples, replace=False):
    r"""
    Uniform sampling over the vertices

    Parameters
    ----------
    graph: graph object
        Must have the number of vertices accessible as an attribute `n_vertices`
    n_samples: int
        Number of measurements to take
    replace: bool
        Whether to sample with replacement (True) or not (False).
        (Default: True)

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
    graph: graph object
        Must have the number of vertices accessible as an attribute `n_vertices`
    n_samples: int
        Number of measurements to take
    replace: bool
        Whether to sample with replacement (True) or not (False).
        (Default: True)

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
    graph: graph object
        Must have the number of vertices accessible as an attribute `n_vertices`
    n_samples: int
        Number of measurements to take
    replace: bool
        Whether to sample with replacement (True) or not (False).
        (Default: True)

    Returns
    -------
    (n_samples, ) array
        Indices of the sampled vertices
        
    """

    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=1./graph.dw,
                              replace=replace)


def inter_comm_degree_vertex(graph, n_samples, labels, replace=False):
    r"""
    Sampling proportional to connection with members of other communities

    Parameters
    ----------
    graph: graph object
        Must have the number of vertices accessible as an attribute `n_vertices`
    n_samples: int
        Number of measurements to take
    labels: array
        Labelling vector encoding to which community each vertex belongs.
    replace: bool
        Whether to sample with replacement (True) or not (False).
        (Default: True)

    Returns
    -------
    (n_samples, ) array
        Indices of the sampled vertices
        
    """

    # Copy the (weigthed) adjacency matrix
    adjacency = sparse.lil_matrix(graph.W.copy())

    # Make sure the labels vector is a numpy array
    labels = np.asarray(labels)

    for vertex in np.arange(len(labels)):
        # Disconnect vertices that belong in the same community
        adjacency[vertex, labels == labels[vertex]] = 0

    inter_comm_degree = np.array(adjacency.sum(axis=1)).flatten()

    return sample_coordinates(graph.n_vertices,
                              n_samples,
                              probs=inter_comm_degree,
                              replace=replace)

