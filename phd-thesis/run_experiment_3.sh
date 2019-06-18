#!/bin/bash

echo "Experiment 3: phase transitions from naive coherence sampling and TV interpolation."
export OMP_NUM_THREADS=1

# swiss-national-council #
echo "swiss-national-council"
python3 pt_swiss_national_council.py -nt 51 -nm 101 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_snc_naive_coherence_samp_tv_interp"

# BSDS300 #
echo "BSDS300"
python3 pt_bsds.py -sub 'both' -gtype 'grid_and_patches' -nt 15 -nm 25 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_bsds300_naive_coherence_samp_tv_interp"

# 2-SSBM #
echo "2-SSBM"
python3 pt_ssbm.py -nv 1000 -nc 2 -nt 25 -b 0.5 -na 51 -nm 51 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_2ssbm_naive_coherence_samp_tv_interp"

# Unbalanced 2-SBM #
echo "Unbalanced 2-SBM"
python3 pt_sbm.py -nv 1000 -nc 2 -nvpc 200 800 -nt 25 -b 0.5 -na 51 -nm 51 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_2sbm_naive_coherence_samp_tv_interp"

# email-EU-core #
echo "email-EU-core"
python3 pt_email_eu_core.py -nc 5 -nt 51 -nm 101 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_email_eu_core_naive_coherence_samp_tv_interp"