#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities module

"""


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
    
    LinOp = LinearOperator(shape=shape, matvec=L, rmatvec=Lt)
    
    return svds(LinOp, k=1, which='LM', return_singular_vectors=False)[0]


