#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Phase transition module

"""


import datetime
import itertools

import numpy as np
import recovery as rec
import sampling as smp
import graphs_signals as gs
import pathos.multiprocessing as mp

from tqdm import tqdm
from utils import save_obj


## MAIN FUNCTIONS ##

def grid_evaluation(param_list_one, param_list_two, param_eval, n_trials=16, 
                    aggr_method=np.mean, save_dir='data/', file_name='grid evaluation',
                    save_to_disk=True, save_each=1000, chunksize=1.):
    r"""
    Evaluates a grid of parameter pairs across repeated trials and aggregates the result.

    Parameters
    ----------
    param_list_one : array_like
        List of values to test for the first parameter.
    param_list_two : array_like
        List of values to test for the second parameter.
    param_eval : callable
        Must take a pair of parameter values and return objects that can be evaluated 
        by `aggr_meth`.
    n_trials : int, optional
        Number of trials to run for each parameter pair. (default is `16`)
    aggr_method : callable, optional
        The aggregation method for the values returned by `patam_eval` on different 
        trials for the same parameter pair. (default is :func:`numpy.mean`)
    save_dir : string, optional
        Directory onto which save the result. (default is 'data/')
    file_name : string, optional
        Optional name for the file. It is always prepended with the time stamp at the 
        end of the grid evaluation. (default is 'grid evaluation')
    save_to_disk : bool, optional
        Whether to save the experiment to disk (True) or not (False). (default is `True`)
    save_each : int, optional
        Save the experiment each time `save_each` grid points are computed. (default is `1000`)
    chunksize : int
        The size of the chunks of jobs sent to each parallel worker. (default is `1`)
        
    Returns
    -------
    

    """
    
    # Assemble product list of parameter values
    params = list(itertools.product(param_list_one, param_list_two))
    n_rows = len(param_list_one)
    n_cols = len(param_list_two)
    n_grid_pts = len(params)
    
    # Define the function to compute for each grid point
    def grid_fun(tup):
        trial_out = np.nan * np.ones((n_trials,))
        for i in np.arange(n_trials):
            trial_out[i] = param_eval(tup[0], tup[1])
        return aggr_method(trial_out)
    
    # Define recording procedure
    def save_experiment(grid):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_path = save_dir + now + ' ' + file_name + '.pkl'
        experiment = {
            'date': now,
            'rows': param_list_one,
            'cols': param_list_two,
            'n_trials': n_trials,
            'grid': np.reshape(grid, (n_rows, n_cols)),
            'path': save_path
         }
        if save_to_disk:
            save_obj(experiment, save_path)
        return experiment
    
    # Set a pool of workers
    nb_workers = mp.cpu_count()
    print('Working with {} processes.'.format(nb_workers))
    pool = mp.Pool(nb_workers)
    
    # Iterate `grid_fun` across workers
    it = pool.imap(grid_fun, params, chunksize=chunksize)
    grid = np.nan * np.ones((n_grid_pts))
    for idx, val in enumerate(tqdm(it, total=n_grid_pts)):
        grid[idx] = val
        
        # Make sure that we save after each couple of iterations
        if (idx >= save_each) and (idx % save_each == 0): 
            experiment = save_experiment(grid)
    
    # Close pool
    pool.close()
    pool.join()
    
    experiment = save_experiment(grid)
    
    return experiment
    
    
## SPECIAL CASES ##

def ssbm(n_vertices=100, n_communities=2, a_list=None, b=.5, m_list=None, 
         smp_design=None, rec_fun=None, file_name='ssbm phase transition', **kwargs):
    r"""
    Compute phase transition of SSBM.
    
    Parameters
    ----------
    n_vertices : int, optional
        Number of vertices in the graph.
    n_communities : int, optional
        Number of communities in the graph. (default is `2`)
    a_list : list of float, optional
        A list of scalars multiplying `np.log(n_vertices)/n_vertices` to yield the 
        intra-community connection probabilities. (default is an equally spaced 
        list in :math:`[0,2k] \pm 2\sqrt{kb} + b`, where :math:`k` is the number 
        of communities)
    b : float, optional
        A scalar multiplying `np.log(n_vertices)/n_vertices` to yield the inter-community 
        connection probabilities. (default is `.5`)
    m_list : list of int, optional
        A list of number of samples to consider. (default is an equally spaced 
        list in :math:`[0,n]`, where :math:`n` is the number of vertices in the graph)
    smp_design : callable, optional
        A sampling design function taking a graph and a number of measurements, and 
        returning the indices of sampled vertices.
    rec_fun : callable, optional
        The function used to recover the subsampled signal. It must take as input a graph,
        a list of sampled vertices, and a list of sampled values.
    file_name : string, optional
        The name to be appended to the current timestamp when saving the experiment file.
        (default is 'ssbm phase transition')
      
    Notes
    -----
    You can also explicitely set the arguments in :func:`grid_evaluation` in this function 
    call.

    """
    
    if a_list is None:
        a_list = np.linspace(-1, 1, 10)
        a_list *= n_communities
        a_list += (np.sqrt(n_communities) + np.sqrt(b)) ** 2.
        
    if m_list is None:
        m_list = np.linspace(0, n_vertices, 10)
        
    if smp_design is None:
        smp_design = lambda g, m: smp.uniform_vertex(g, m, replace=True)
        
    if rec_fun is None:
        rec_fun = lambda g, s_ver, s_val: rec.tv_interpolation(g, s_ver, s_val,
                                                               rtol=1e-6 * n_vertices**(-1/2),
                                                               maxit=5000,
                                                               verbosity='NONE')
        
    def param_eval(a, m):
        # Draw a graph
        graph, indicator_vectors = gs.ssbm(n_vertices=n_vertices, 
                                           n_communities=n_communities, 
                                           a=a, 
                                           b=b)
        # Set label signal to be recovered
        labels = indicator_vectors[-1,:]
        
        # Subsample
        sampled_vertices = smp_design(graph, m)
        sampled_values = labels[sampled_vertices]
        
        # Recover
        recovered_signal = rec_fun(graph, sampled_vertices, sampled_values)
        
        # Measure the error
        rel_err = np.linalg.norm(recovered_signal - labels, ord=2) / np.linalg.norm(labels, ord=2)
        
        return rel_err
    
    experiment = grid_evaluation(param_list_one=a_list, 
                           param_list_two=m_list, 
                           param_eval=param_eval,
                           file_name=file_name,
                           **kwargs)
    
    experiment['b'] = b
    experiment['row_label'] = 'a'
    experiment['col_label'] = 'm'
    experiment['n_vertices'] = n_vertices
    experiment['n_communities'] = n_communities
    
    return experiment