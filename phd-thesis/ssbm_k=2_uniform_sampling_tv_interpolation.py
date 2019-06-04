#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""2-SSBM phase transition with uniform sampling and TV interpolation.

"""

import os, sys
cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import utils
import datetime

import numpy as np
import recovery as rec
import sampling as smp
import graphs_signals as gs
import phase_transition as pt


if __name__ == "__main__":
    
    # Hyperparameters
    n_vertices = 1000 # Number of vertices 
    n_communities = 2 # Number of communities
    b = 0.5           # Factor in the inter-community connection probabilities
    n_trials = 100    # Number of trials per grid point
    file_name = 'pt ssbm k=2 uniform tv' # File name
    
    # Factors in the intra-community connection probabilities
    a_list = np.linspace(-1, 1, 100)
    a_list *= np.sqrt(n_communities)
    a_list += np.sqrt(n_communities) + np.sqrt(b)
    
    # List of numbers of measurements
    m_list = np.linspace(0, n_vertices, 100)
    
    # Sampling design
    smp_design = lambda g, m: smp.uniform_vertex(g, m, replace=True)
    
    # Recovery function
    rec_fun = lambda g, s_ver, s_val: rec.tv_interpolation(g, s_ver, s_val,
                                                           rtol=1e-6 * n_vertices**(-1/2),
                                                           maxit=5000,
                                                           verbosity='NONE')
    
    # Parameter evaluation function
    def param_eval(a, m):
        # Draw a graph
        graph, indicator_vectors = gs.ssbm(n_vertices=n_vertices, 
                                           n_communities=n_communities, 
                                           a=a, 
                                           b=b)
        # Set signal to be recovered
        gt_signal = indicator_vectors[-1,:]
        
        _, rel_err = utils.standard_pipeline(graph, 
                                             gt_signal, 
                                             m,
                                             smp_design, 
                                             rec_fun)
        
        return rel_err
    
    experiment = pt.grid_evaluation(param_list_one=a_list, 
                                    param_list_two=m_list, 
                                    param_eval=param_eval,
                                    file_name=file_name,
                                    aggr_method=np.median,
                                    n_trials=n_trials,
                                    save_to_disk=True)
    
    experiment['b'] = b
    experiment['row_label'] = 'a'
    experiment['col_label'] = 'm'
    experiment['n_vertices'] = n_vertices
    experiment['n_communities'] = n_communities
    experiment['sampling_design'] = 'uniform_vertex'
    experiment['recovery_function'] = 'tv_interpolation'
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = 'data/' + now + ' ' + file_name + '.pkl'
    utils.save_obj(experiment, save_path)