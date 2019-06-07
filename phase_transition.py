#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Phase transition module

"""


import utils
import datetime
import itertools

import numpy as np
import recovery as rec
import sampling as smp
import graphs_signals as gs
import pathos.multiprocessing as mp

from tqdm import tqdm

#import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')


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
    param_list_two : array_like, optional
        List of values to test for the second parameter. Can be empty, in which case a 
        one-dimensional grid is evaluated.
    param_eval : callable
        Must take an instance of parameter values and return an object that can be evaluated 
        by `aggr_meth`. It should accept one input if `param_list_two` is empty, and two inputs 
        otherwise.
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
    dict
        A dictionary with the results of the experiment.

    """
    
    
    if not list(param_list_two): # If `param_list_two` is empty
        params = param_list_one
        grid_shape = (len(param_list_one),)
        is_really_grid = False
    
    else:
        params = list(itertools.product(param_list_one, param_list_two))
        grid_shape = (len(param_list_one), len(param_list_two))
        is_really_grid = True
        
    def grid_fun(point): # Function to compute for each grid point
        
        trial_out = np.nan * np.ones((n_trials,))
        
        for i in np.arange(n_trials):
           
            if is_really_grid:
                trial_out[i] = param_eval(point[0], point[1])
            else: # If `param_list_two` is empty
                trial_out[i] = param_eval(point)
            
        return aggr_method(trial_out)
        
    n_grid_pts = len(params)
    
    # Recording procedure
    def record_experiment(grid):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_path = save_dir + now + ' ' + file_name + '.pkl'
        experiment = {
            'date': now,
            'rows': param_list_one,
            'cols': param_list_two,
            'n_trials': n_trials,
            'grid': np.reshape(grid, grid_shape),
            'path': save_path
         }
        if save_to_disk:
            utils.save_obj(experiment, save_path)
        return experiment
    
    # Set a pool of workers
    nb_workers = mp.cpu_count()
    print('Working with {} processes.'.format(nb_workers))
    pool = mp.Pool(nb_workers)
    
    # Iterate `grid_fun` across workers
    it = pool.imap(grid_fun, params, chunksize=chunksize)
    grid = np.nan * np.ones((n_grid_pts,))

    for idx, val in enumerate(tqdm(it, total=n_grid_pts)):
        grid[idx] = val
        
        # Make sure that we save after each couple of iterations
        if (idx >= save_each) and (idx % save_each == 0): 
            experiment = record_experiment(grid)
    
    # Close pool
    pool.close()
    pool.join()
    
    experiment = record_experiment(grid)
    
    return experiment


def line_evaluation(param_list, param_eval, file_name='line evaluation', **kwargs):
    r"""
    Evaluates a list of parameter pairs across repeated trials and aggregates the result.

    Parameters
    ----------
    param_list : array_like
        List of values to test for parameter of interest.
    param_eval : callable
        Must take a parameter instance and return an object that can be evaluated 
        by `aggr_meth` (see :func:`grid_evaluation`).
    file_name : string, optional
        Optional name for the file. (default is 'line evaluation')
        
    Returns
    -------
    dict
        A dictionary with the results of the experiment.
        
    Notes
    -----
    You can also explicitely set the arguments in :func:`grid_evaluation` in this function 
    call.

    """
    
    experiment = grid_evaluation(param_list_one=param_list,
                                 param_list_two=[],
                                 param_eval=param_eval,
                                 file_name=file_name,
                                 **kwargs)

    experiment['line'] = experiment.pop('grid')
    experiment['cols'] = experiment.pop('rows')
    
    return experiment
