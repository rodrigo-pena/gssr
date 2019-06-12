# -*- coding: utf-8 -*-

"""swiss-national-council phase transition

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

from argparse import ArgumentParser


if __name__ == "__main__":
    
    # Parse arguments
    parser = ArgumentParser(description='Error line for class label recovery in swiss-nacional-council,'
                            + ' varying the number of label measurements.')
    
    parser.add_argument('-p', action='store', nargs='?', default='../data/swiss-national-council/', type=str, 
                        help="path to the folder containing the swiss-nacional-council data" )
    parser.add_argument('-nt', action='store', nargs='?', default=5, type=int, 
                        help='number of random trials for each grid point')
    parser.add_argument('-nm', action='store', nargs='?', default=11, type=int, 
                        help='number of points in the measurements axis')
    parser.add_argument('-sd', action='store', nargs='?', default='uniform_vertex', 
                        type=str, choices=['uniform_vertex',
                                          'inv_degree_vertex'],
                        help='vertex sampling design')
    parser.add_argument('-rf', action='store', nargs='?', default='tv_interpolation', 
                        type=str, choices=['tv_interpolation', 
                                           'tv_least_sq', 
                                           'dirichlet_form_interpolation', 
                                           'dirichlet_form_least_sq'],
                        help='recovery function')
    parser.add_argument('-fn', action='store', nargs='?', default='pt_snc', 
                        type=str,
                        help='output file name suffix')
    
    args = parser.parse_args()
    
    # Draw graph and indicator vectors
    nn_params = {'NNtype': 'knn',
                 'use_flann': True,
                 'center': False,
                 'rescale': True,
                 'k': 25,
                 'dist_type': 'euclidean'}
    graph, indicator_vectors = gs.swiss_national_council(path=args.p, **nn_params)
    args.nv = graph.n_vertices
    
    # Set signal to be recovered
    parties = graph.info['parties']
    parties_in_gt = ['UDC', 'FDP']
    cls_mask = [False] * len(parties)
    for p in parties_in_gt:
        cls_mask += (parties == p)
        
    gt_signal = np.sum(indicator_vectors[cls_mask,:], axis=0)
    
    n_communities = 2
    comm_sizes = [int(np.sum(gt_signal)), int(args.nv - np.sum(gt_signal))]
    
    # List of parameters in the horizontal axis of the grid
    # (Number of measurements)
    list_m = np.linspace(0, args.nv, args.nm)
    
    # Sampling design
    smp_design = utils.select_sampling_design(args.sd, replace=True)
    
    # Recovery function
    rec_fun = utils.select_recovery_function(args.rf, 
                                             rtol=1e-6 * (args.nv ** (-1/2)),
                                             maxit=5000,
                                             verbosity='NONE')
    
    # Parameter evaluation function
    def param_eval(m): 
        _, rel_err = utils.standard_pipeline(graph, 
                                             gt_signal, 
                                             m,
                                             smp_design, 
                                             rec_fun)
        return rel_err
    
    
    # Parameter line evaluation
    experiment = pt.line_evaluation(param_list=list_m,
                                    param_eval=param_eval,
                                    file_name=args.fn,
                                    aggr_method=np.median,
                                    n_trials=args.nt,
                                    save_to_disk=True)
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = 'data/' + now + ' ' + args.fn + '.pkl'
    
    
    # Update fields
    experiment['date'] = now
    experiment['n_vertices'] = args.nv
    experiment['n_communities'] = n_communities
    experiment['comm_sizes'] = comm_sizes
    experiment['parties_in_gt'] = parties_in_gt
    experiment['gt_signal'] = gt_signal
    experiment['cols'] = list_m
    experiment['col_label'] = 'm'
    experiment['col_tick_labels'] = [
        r"$0$",   # Start
        r"$n/2$", # Mid
        r"$n$"    # End
    ]
    experiment['sampling_design'] = args.sd
    experiment['recovery_function'] = args.rf
    experiment['path'] = save_path
    experiment['nn_params'] = nn_params
    
    
    # Save
    utils.save_obj(experiment, save_path)