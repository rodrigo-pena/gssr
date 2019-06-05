#!/bin/bash

echo "Running all experiments. This might take a long time..."

# Phase transition: 2-SSBM with uniform sampling and TV recovery
python3 pt_ssbm.py -nv 1000 -nc 2 -nt 25 -b 0.5 -na 101 -nm 101 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_2ssbm_unif_samp_tv_interp"