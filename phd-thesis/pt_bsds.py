# -*- coding: utf-8 -*-

"""BSDS300 phase transition

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
    parser = ArgumentParser(description='Error lines for class label recovery in BSDS300,'
                            + ' varying number of label measurements.')
    
    parser.add_argument('-p', action='store', nargs='?', default='../data/BSDS300/', type=str, 
                        help="path to the folder containing the BSDS300 data" )
    parser.add_argument('-sub', action='store', nargs='?', default='both', type=str, 
                        choices=['train', 'test', 'both'],
                        help="subset of images to use" )
    parser.add_argument('-gtype', action='store', nargs='?', default='grid_and_patches', 
                        type=str, choices=['grid', 'patches', 'grid_and_patches'],
                        help="type of graph to construct for each image" )
    parser.add_argument('-nt', action='store', nargs='?', default=5, type=int, 
                        help='number of random trials for each grid point')
    parser.add_argument('-nm', action='store', nargs='?', default=11, type=int, 
                        help='number of points in the measurements axis')
    parser.add_argument('-sd', action='store', nargs='?', default='uniform_vertex', 
                        type=str, choices=['uniform_vertex',
                                           'naive_tv_coherence',
                                           'jump_set_tv_coherence'],
                        help='vertex sampling design')
    parser.add_argument('-rf', action='store', nargs='?', default='tv_interpolation', 
                        type=str, choices=['tv_interpolation', 
                                           'dirichlet_form_interpolation'],
                        help='recovery function')
    parser.add_argument('-fn', action='store', nargs='?', default='pt_bsds300', 
                        type=str,
                        help='output file name suffix')
    
    args = parser.parse_args()
    
    # Get image IDs
    list_id = utils.get_bsds300_id_list(path=args.p, subset=args.sub)
    
    # List of parameters in the horizontal axis of the grid
    # (Number of measurements)
    list_m = np.linspace(0, 1, args.nm) # Relative to the number of pixels in each image
    
    # Parameter evaluation function
    def param_eval(idx, m): 
        
        # Draw graph and indicator vectors
        graph, _ = gs.bsds300(idx, 
                              path=args.p, 
                              graph_type=args.gtype, 
                              k=3, 
                              use_flann=True)
    
        # Set signal to be recovered
        gt_signal = graph.info['node_com'] # Segmentation labels
        
        # Sampling design
        smp_design = utils.select_sampling_design(args.sd, 
                                                  gt_signal, 
                                                  replace = True)
        
        # Recovery function
        rec_fun = utils.select_recovery_function(args.rf, 
                                                 rtol=1e-6 * (graph.n_vertices ** (-1/2)),
                                                 maxit=5000,
                                                 verbosity='NONE')
        
        _, rel_err = utils.standard_pipeline(graph, 
                                             gt_signal, 
                                             m * graph.n_vertices, # Scale the relative size
                                             smp_design, 
                                             rec_fun)
        return rel_err
    
    
    # Parameter grid evaluation
    experiment = pt.grid_evaluation(param_list_one=list_id,
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
    experiment['graph_type'] = args.gtype
    experiment['rows'] = list_id
    experiment['row_label'] = 'Image ID'
    experiment['row_tick_labels'] = []
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
    
    
    # Save
    utils.save_obj(experiment, save_path)