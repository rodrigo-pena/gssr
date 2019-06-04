#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities module

"""

import pickle

import numpy as np


def interpolation_projection(signal, sampled_vertices, sampled_values):
    r"""
    Orthogonal projection of a vector onto the interpolation set.
    """
    
    projection = np.copy(signal)
    projection[sampled_vertices] = sampled_values
    return projection


def sampling_restriction(signal, sampled_vertices):
    r"""
    Restrict a vector to the coordinates contained in the sampling set.
    """
    return signal[sampled_vertices]

def sampling_embedding(embedding_dim, sampled_values, sampled_vertices):
    r"""
    Embed a sampling-restricted vector into a higher-dimension ambient space.
    """
    embedding = np.zeros((embedding_dim,))
    embedding[sampled_vertices] = sampled_values
    return embedding

def nan_off_sample(n_vertices, sampled_vertices, sampled_values):
    r"""
    Insert `np.nan` values at the un-sampled coordinates.
    """
    sampled_signal_with_nan = np.nan * np.ones(n_vertices,)
    sampled_signal_with_nan[sampled_vertices] = sampled_values
    return sampled_signal_with_nan

def save_obj(obj, path):
    r"""
    Save Python object to disk as a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    r"""
    Load pickle file from disk.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def get_diff_op(graph):
    r"""
    Get graph differential operators.
    
    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
        The graph.
        
    """
    
    graph.compute_laplacian(lap_type='combinatorial')
    graph.compute_differential_operator()
    op_direct = lambda z: graph.grad(z) # Graph gradient (incidence transposed)
    op_adjoint = lambda z: graph.div(z) # Graph divergent (incidence matrix)
    with np.errstate(divide='ignore',invalid='ignore'):
        graph.estimate_lmax()
    op_specnorm = np.sqrt(graph.lmax)
    return op_direct, op_adjoint, op_specnorm

def spectral_norm(shape, L, Lt):
    r"""
    Estimate largest singular value of L using ARPACK as an eigensolver.

    Parameters
    ----------
    shape : tuple
        Dimensions of the linear map (dim(range L), dim(dom L)).
    L : callable
        A function representing a linear mapping between vector spaces.
    Lt : callable
        A function representing the adjoint linear mapping.

    Returns
    -------
    Largest singular value of L.

    Notes
    -----
    This function can be unstable and diverge.

    """
    
    from scipy.sparse.linalg import LinearOperator, svds
    
    lin_op = LinearOperator(shape=shape, matvec=L, rmatvec=Lt)
    
    try:
        spec_norm = svds(lin_op, k=1, which='LM', 
                         return_singular_vectors=False)[0]
    except:
        raise ValueError('The spectral norm estimate did not converge')
    
    return spec_norm
