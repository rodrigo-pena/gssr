#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""2-SSBM phase transition with uniform sampling and TV interpolation.

"""

import os, sys
cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import datetime

import numpy as np
import recovery as rec
import sampling as smp
import graphs_signals as gs
import phase_transition as pt

from utils import save_obj

if __name__ == "__main__":
    
    # Number of vertices and communities
    n_vertices = 100
    n_communities = 2
    
    # Scalar multiplying `np.log(n_vertices)/n_vertices` to yield the 
    # inter-community connection probabilities
    b = 0.5
    
    # A list of scalars multiplying `np.log(n_vertices)/n_vertices` to yield the 
    # intra-community connection probabilities
    a_list = np.linspace(-1, 1, 10)
    a_list *= np.sqrt(n_communities)
    a_list += np.sqrt(n_communities) + np.sqrt(b)
    
    # List of numbers of measurements
    m_list = np.linspace(0, n_vertices, 10)
    
    # Sampling design
    smp_design = lambda g, m: smp.uniform_vertex(g, m, replace=True)
    
    # Recovery function
    rec_fun = lambda g, s_ver, s_val: rec.tv_interpolation(g, s_ver, s_val,
                                                           rtol=1e-6 * n_vertices**(-1/2),
                                                           maxit=5000,
                                                           verbosity='NONE')
    # Number of trials per grid point
    n_trials = 10

    experiment = pt.ssbm(n_vertices=n_vertices, 
                n_communities=n_communities, 
                a_list=a_list, 
                b=b, 
                m_list=m_list, 
                smp_design=smp_design, 
                rec_fun=rec_fun,
                aggr_method=np.median,
                n_trials=n_trials,
                save_to_disk=True,
                chunksize=1)
    
    experiment['sampling_design'] = 'uniform_vertex'
    experiment['recovery_function'] = 'tv_interpolation'
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = 'data/' + now + ' ' + 'pt ssbm k=2 uniform tv' + '.pkl'
    save_obj(experiment)