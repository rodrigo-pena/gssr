# -*- coding: utf-8 -*-

"""Utilities module

"""

import pickle

import numpy as np


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
    with np.errstate(divide='ignore', invalid='ignore'):
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


def select_recovery_function(name, **kwargs):
    r"""
    Use a string to select sampling designs in `recovery`.
    """

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


def select_sampling_design(name, *args, **kwargs):
    r"""
    Use a string to select sampling designs in `sampling`.
    """

    import sampling as smp

    if name == 'uniform_vertex':
        return lambda g, m: smp.uniform_vertex(g, m, **kwargs)

    elif name == 'naive_tv_coherence':
        return lambda g, m: smp.naive_tv_coherence(g, m, **kwargs)

    elif name == 'jump_set_tv_coherence':
        return lambda g, m: smp.jump_set_tv_coherence(g, m, *args, **kwargs)

    else:
        raise ValueError("There is no sampling design with this name.")


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


## BSDS300 ##

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


## swiss-national-council ##

def get_parliament_quadrant_coordinates(n_chairs, quadrant_angle=np.pi/4, flipped=False):
    r"""
    Get coordinates in analogy to the positions of the chairs on a quadrant of the Swiss parliament.

    Parameters
    ----------
    n_chairs : int
        Number of chairs in the quadrant.
    quadrant_angle : float, optional
        The angle that the quadrant slice makes with the x axis. (default is `numpy.pi/4`)
    flipped : bool
        Draw the quadrant flipped. (default is `False`)

    Returns
    -------
    (n_chairs, 2) numpy.ndarray
        The coordinates of each of the quadrant's chair on the 2D plane.

    Notes
    -----
    This helper function is used in :func:`get_parliament_coordinates()`.

    """
    radius = 1.  # Distance of the first chair to the origin of the 2d-plane
    count = 0  # Number of chairs added so far
    n_chairs_in_row = 2  # Number of chairs to add in the first row
    coords = np.zeros((n_chairs, 2))  # List of coordinates of all the chairs in the quadrant

    if flipped:
        start_angle, end_angle = (quadrant_angle, 0)
    else:
        start_angle, end_angle = (0, quadrant_angle)

    while count < n_chairs:

        # Angles of the row's chairs w.r.t. the x axis
        angles = np.linspace(start_angle, end_angle, n_chairs_in_row)

        # Coordinates of the row's chairs in the 2d-plane
        row_coords = radius * np.array(list(zip(np.cos(angles), np.sin(angles))))

        # Gather coordinates in the list
        if count + n_chairs_in_row > n_chairs:
            end_idx = (n_chairs - count)
            coords[count:, :] = row_coords[-end_idx:, :]
        else:
            end_idx = -1
            coords[count:count+n_chairs_in_row, :] = row_coords

        # Update parameters for the next row
        radius += radius / (1 + radius)
        count += n_chairs_in_row
        n_chairs_in_row += 2
        #start_angle, end_angle = (end_angle, start_angle)

    return coords


def get_parliament_coordinates(n_councillors):
    r"""
    Get coordinates in analogy to the positions of the chairs in the Swiss parliament.

    Parameters
    ----------
    n_councillors : int
        Number of councillors to fir in the parliament

    Returns
    -------
    (n_councillors, 2) numpy.ndarray
        The coordinates of each councillor on the 2D plane

    Notes
    -----
    This helper function is used in :func:`graphs_signals.swiss_national_council()` to
    set the coordinates of the vertices of the graph.

    """

    # The parliament has 200 chairs
    n_chairs = 200
    # List of coordinates of the council members in the parliament
    coords = np.zeros((n_councillors, 2))
    # Rotation angles of the quadrants
    quadrant_angles = np.pi * np.array([0, 1/4, 2/4, 3/4, 1])
    # Radial translation of the quadrants
    center_vec = np.array([np.cos(np.pi/8), np.sin(np.pi/8)])
    # Number of chairs per quadrant
    n_chairs_per_quadrant = np.floor(200/4).astype(int)
    # Remaining councillors
    n_remaining_councillors = n_councillors - n_chairs

    count = 0 # Chair count
    flipped = False # Flip quadrant

    # Gather the coordinates of each quadrant
    for y, x in list(zip(np.sin(quadrant_angles), np.cos(quadrant_angles))):

        if count + n_chairs_per_quadrant <= 200:
            n_chairs_to_add = n_chairs_per_quadrant
        else:
            n_chairs_to_add = n_remaining_councillors

        # Define a rotation matrix
        R = np.array(((x, -y), (y, x)))

        # Get quadrant coordinates
        q_coords = get_parliament_quadrant_coordinates(n_chairs_to_add,
                                                       flipped=flipped)
        # Translate and rotate
        q_coords = (R @ (q_coords + center_vec).T).T

        # Update list of coordinates
        if flipped:
            coords[count:count+n_chairs_to_add, :] = q_coords[-1::-1]
        else:
            coords[count:count+n_chairs_to_add, :] = q_coords

        # Update parameters for next iteration
        count += n_chairs_per_quadrant
        flipped = not flipped

    return coords


