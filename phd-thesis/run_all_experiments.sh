#!/bin/bash

echo "Running all experiments. This will take a long time. You could also try running line by line."

## 1. Phase transition: Uniform sampling and TV recovery ##

# 2-SSBM #
#python3 pt_ssbm.py -nv 1000 -nc 2 -nt 25 -b 0.5 -na 101 -nm 101 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_2ssbm_unif_samp_tv_interp"

# Remark: It turns out that the resolution is too high for the behaviors observed in the plot,
# so the rest of the experiments are done with a smaller resolution.

# Unbalanced 2-SBM #
#python3 pt_sbm.py -nv 1000 -nc 2 -nvpc 200 800 -nt 25 -b 0.5 -na 51 -nm 51 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_2sbm_unif_samp_tv_interp"

# email-EU-core #
#python3 pt_email_eu_core.py -nc 5 -nt 51 -nm 101 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_email_eu_core_unif_samp_tv_interp"

# swiss-national-council #
python3 pt_swiss_national_council.py -nt 51 -nm 101 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_snc_unif_samp_tv_interp"

# BSDS300 #
#python3 pt_bsds.py -sub 'both' -gtype 'grid_and_patches' -nt 15 -nm 25 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_bsds300_unif_samp_tv_interp"

