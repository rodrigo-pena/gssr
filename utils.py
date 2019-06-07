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

def standard_pipeline(graph, gt_signal, m, smp_design, rec_fun):
    r"""
    Standard pipeline for subsampling and recovery.
    
    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
        The graph.
    gt_signal : `numpy.ndarray`
        The ground-truth signal to be recovered.
    m_list : int
        Number of samples (measurements).
    smp_design : callable
        A sampling design function taking a graph and a number of measurements, and 
        returning the indices of sampled vertices.
    rec_fun : callable
        The function used to recover the subsampled signal. It must take as input a graph,
        a list of sampled vertices, and a list of sampled values.
        
    Returns
    -------
    `numpy.ndarray`
        The recovered signal.
    float
        The recovery error.
        
    """
    # Subsample
    sampled_vertices = smp_design(graph, m)
    sampled_values = gt_signal[sampled_vertices]

    # Recover
    recovered_signal = rec_fun(graph, sampled_vertices, sampled_values)
    
    # Measure the error
    rel_err = np.linalg.norm(recovered_signal - gt_signal, ord=2) 
    rel_err /= np.linalg.norm(gt_signal, ord=2)
    
    return recovered_signal, rel_err

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


def get_bsds300_id_list(path='data/BSDS300/', subset='train'):
    r"""
    Get list of image IDs in the BSDS300 dataset.
    
    Parameters
    ----------
    path : str, optional
        The path to the directory containing the BSDS300 dataset. 
        (default is 'data/BSDS300/')
    subset : str, optional
        Image subset to use. Options are 'train', 'test' or 'both'. 
        (default is 'train')
        
    Returns
    -------
    list of str
        List of image IDs in the chosen subset.
        
    """
    
    ids_train = [line.rstrip('\n') for line in open(path + 'iids_train.txt')]
    ids_test = [line.rstrip('\n') for line in open(path + 'iids_test.txt')]
    
    if subset == 'train':
        return ids_train
    elif subset == 'test':
        return ids_test
    elif subset == 'both':
        return ids_train + ids_test
    else:
        raise ValueError("Valid options for subset are 'train', 'test' or 'both'.")
        
        
def get_bsds300_subset(img_id, path='data/BSDS300/'):
    r"""
    Get BSDS300 image subset from image ID.
    
    Parameters
    ----------
    img_id : str
        ID of the image.
    path : str, optional
        The path to the directory containing the BSDS300 dataset. 
        (default is 'data/BSDS300/')
    subset : str, optional
        Image subset to use. Options are 'train', 'test' or 'both'. 
        (default is 'train')
        
    Returns
    -------
    str
        'train', 'test' or 'unknown'.
        
    """
    
    ids_train = [line.rstrip('\n') for line in open(path + 'iids_train.txt')]
    ids_test = [line.rstrip('\n') for line in open(path + 'iids_test.txt')]
    
    if img_id in ids_train:
        return 'train'
    elif img_id in ids_test:
        return 'test'
    else:
        raise 'unknown'
      
    
def select_recovery_function(name, **kwargs):
    
    import recovery as rec
    
    if name == 'tv_interpolation':
        return lambda g, s_ver, s_val: rec.tv_interpolation(g, s_ver, s_val, **kwargs)

    elif name == 'tv_least_sq':
        return lambda g, s_ver, s_val: rec.tv_least_sq(g, s_ver, s_val, **kwargs)

    elif name == 'dirichlet_form_interpolation':
        return lambda g, s_ver, s_val: rec.dirichlet_form_interpolation(g, s_ver, s_val, **kwargs)

    elif name == 'dirichlet_form_least_sq':
        return lambda g, s_ver, s_val: rec.dirichlet_form_least_sq(g, s_ver, s_val, **kwargs)
    
    else:
        raise ValueError("There is no recovery function with this name.")

    
def select_sampling_design(name, **kwargs):
    
    import sampling as smp
        
    if name == 'uniform_vertex':
        return lambda g, m: smp.uniform_vertex(g, m, **kwargs)
    
    elif name == 'inv_degree_vertex':
        return lambda g, m: smp.inv_degree_vertex(g, m, **kwargs)
    
    else:
        raise ValueError("There is no sampling design with this name.")
