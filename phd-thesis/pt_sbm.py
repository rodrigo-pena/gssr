# -*- coding: utf-8 -*-

"""k-SBM phase transition

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
    parser = ArgumentParser(description='Error grid for class label recovery in k-SBM,'
                            + ' varying the intra-class connection probability and'
                            + ' the number of label measurements.')
    
    parser.add_argument('-nv', action='store', nargs='?', default=100, type=int, 
                        help='number of vertices in the graph')
    parser.add_argument('-nc', action='store', nargs='?', default=2, type=int, 
                        help='number of classes (communities)')
    parser.add_argument('-nvpc', action='store', nargs='*', default=[20, 80], type=int, 
                        help='number of vertices in each class')
    parser.add_argument('-nt', action='store', nargs='?', default=5, type=int, 
                        help='number of random trials for each grid point')
    parser.add_argument('-b', action='store', nargs='?', default=0.5, type=float, 
                        help='factor in the inter-class connection probabilities')
    parser.add_argument('-na', action='store', nargs='?', default=11, type=int, 
                        help='number of points in the intra-class connection probability axis')
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
    parser.add_argument('-fn', action='store', nargs='?', default='pt_sbm', 
                        type=str,
                        help='output file name suffix')
    
    args = parser.parse_args()
    
    
    # List of parameters in the vertical axis of the grid
    # (Intra-communitity connection probability factor)
    list_a = np.linspace(0, 
                         2. * ((np.sqrt(args.nc) + np.sqrt(args.b))**2. - (args.nc + args.b)),
                         num=args.na)
    list_a += args.nc + args.b
    
    
    # List of parameters in the horizontal axis of the grid
    # (Number of measurements)
    list_m = np.linspace(0, args.nv, args.nm)
    
    
    # Sampling design
    replace = True
    if args.sd == 'uniform_vertex':
        smp_design = lambda g, m: smp.uniform_vertex(g, m, replace=replace)
    elif args.sd == 'inv_degree_vertex':
        smp_design = lambda g, m: smp.inv_degree_vertex(g, m, replace=replace)
        
    
    # Recovery function
    rtol = 1e-6 * (args.nv ** (-1/2))
    maxit = 5000
    verbosity = 'NONE'
    if args.rf == 'tv_interpolation':
        rec_fun = lambda g, s_ver, s_val: rec.tv_interpolation(g, s_ver, s_val,
                                                               rtol=rtol, maxit=maxit,
                                                               verbosity=verbosity)
    elif args.rf == 'tv_least_sq':
        rec_fun = lambda g, s_ver, s_val: rec.tv_least_sq(g, s_ver, s_val,
                                                          rtol=rtol, maxit=maxit,
                                                          verbosity=verbosity)
    elif args.rf == 'dirichlet_form_interpolation':
        rec_fun = lambda g, s_ver, s_val: rec.dirichlet_form_interpolation(g, s_ver, s_val,
                                                                           rtol=rtol, maxit=maxit,
                                                                           verbosity=verbosity)
    elif args.rf == 'dirichlet_form_least_sq':
        rec_fun = lambda g, s_ver, s_val: rec.dirichlet_form_least_sq(g, s_ver, s_val,
                                                                      rtol=rtol, maxit=maxit,
                                                                      verbosity=verbosity)
    
    
    # Parameter evaluation function
    def param_eval(a, m):
        
        # Connection probabilities
        p = a * np.log(args.nv) / args.nv
        q = args.b * np.log(args.nv) / args.nv
        
        # Draw a graph
        graph, indicator_vectors = gs.sbm(n_vertices=args.nv, 
                                          n_communities=args.nc, 
                                          n_vert_per_comm=args.nvpc, 
                                          intra_comm_prob=p,
                                          inter_comm_prob=q)
        
        # Set signal to be recovered
        gt_signal = indicator_vectors[0,:]
        
        _, rel_err = utils.standard_pipeline(graph, 
                                             gt_signal, 
                                             m,
                                             smp_design, 
                                             rec_fun)
        
        return rel_err
    
    
    # Parameter grid evaluation
    experiment = pt.grid_evaluation(param_list_one=list_a, 
                                    param_list_two=list_m, 
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
    experiment['n_communities'] = args.nc
    experiment['n_vertices_per_community'] = args.nvpc
    experiment['b'] = args.b
    experiment['rows'] = list_a
    experiment['row_label'] = 'a'
    experiment['row_tick_labels'] = [
        r"$k + b$",                                                        # Start
        r"$\left( \sqrt{k} + \sqrt{b} \right)^2$",                         # Mid
        r"$2\left( \sqrt{k} + \sqrt{b} \right)^2 - \left( k + b \right)$"  # End
    ]
    experiment['cols_'] = list_m
    experiment['col_label'] = 'm'
    experiment['col_tick_labels'] = [
        r"$0$",   # Start
        r"$n/2$", # Mid
        r"$n$"    # End
    ]
    experiment['sampling_design'] = args.sd
    experiment['recovery_function'] = args.rf
    experiment['path'] = save_path
    
    
    # Save
    utils.save_obj(experiment, save_path)