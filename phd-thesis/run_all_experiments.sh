#!/bin/bash

echo "Running all experiments. This will take a long time. You could also try running line by line."

## 1. Phase transition: Uniform sampling and TV recovery ##
echo "1. Phase transitions: uniform sampling and TV interpolation"

# 2-SSBM #
echo "2-SSBM"
#python3 pt_ssbm.py -nv 1000 -nc 2 -nt 25 -b 0.5 -na 51 -nm 51 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_2ssbm_unif_samp_tv_interp"

# Unbalanced 2-SBM #
echo "Unbalanced 2-SBM"
#python3 pt_sbm.py -nv 1000 -nc 2 -nvpc 200 800 -nt 25 -b 0.5 -na 51 -nm 51 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_2sbm_unif_samp_tv_interp"

# email-EU-core #
echo "email-EU-core"
#python3 pt_email_eu_core.py -nc 5 -nt 51 -nm 101 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_email_eu_core_unif_samp_tv_interp"

# swiss-national-council #
echo "swiss-national-council"
#python3 pt_swiss_national_council.py -nt 51 -nm 101 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_snc_unif_samp_tv_interp"

# BSDS300 #
echo "BSDS300"
#python3 pt_bsds.py -sub 'both' -gtype 'grid_and_patches' -nt 15 -nm 25 -sd "uniform_vertex" -rf "tv_interpolation" -fn "pt_bsds300_unif_samp_tv_interp"


## 2. Phase transition: Uniform sampling and Dirichlet recovery ##
echo "2. Phase transitions: uniform sampling and Dirichlet interpolation"

# 2-SSBM #
echo "2-SSBM"
#python3 pt_ssbm.py -nv 1000 -nc 2 -nt 25 -b 0.5 -na 51 -nm 51 -sd "uniform_vertex" -rf "dirichlet_form_interpolation" -fn "pt_2ssbm_unif_samp_dirichlet_interp"

# Unbalanced 2-SBM #
echo "Unbalanced 2-SBM"
#python3 pt_sbm.py -nv 1000 -nc 2 -nvpc 200 800 -nt 25 -b 0.5 -na 51 -nm 51 -sd "uniform_vertex" -rf "dirichlet_form_interpolation" -fn "pt_2sbm_unif_samp_dirichlet_interp"

# email-EU-core #
echo "email-EU-core"
#python3 pt_email_eu_core.py -nc 5 -nt 51 -nm 101 -sd "uniform_vertex" -rf "dirichlet_form_interpolation" -fn "pt_email_eu_core_unif_samp_dirichlet_interp"

# swiss-national-council #
echo "swiss-national-council"
#python3 pt_swiss_national_council.py -nt 51 -nm 101 -sd "uniform_vertex" -rf "dirichlet_form_interpolation" -fn "pt_snc_unif_samp_dirichlet_interp"

# BSDS300 #
echo "BSDS300"
#python3 pt_bsds.py -sub 'both' -gtype 'grid_and_patches' -nt 15 -nm 25 -sd "uniform_vertex" -rf "dirichlet_form_interpolation" -fn "pt_bsds300_unif_samp_dirichlet_interp"


## 3. Phase transition: Naive coherence sampling and TV recovery ##
echo "3. Phase transitions: naive coherence sampling and TV interpolation"

# 2-SSBM #
echo "2-SSBM"
python3 pt_ssbm.py -nv 1000 -nc 2 -nt 25 -b 0.5 -na 51 -nm 51 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_2ssbm_naive_coherence_samp_tv_interp"

# Unbalanced 2-SBM #
echo "Unbalanced 2-SBM"
python3 pt_sbm.py -nv 1000 -nc 2 -nvpc 200 800 -nt 25 -b 0.5 -na 51 -nm 51 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_2sbm_naive_coherence_samp_tv_interp"

# email-EU-core #
echo "email-EU-core"
python3 pt_email_eu_core.py -nc 5 -nt 51 -nm 101 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_email_eu_core_naive_coherence_samp_tv_interp"

# swiss-national-council #
echo "swiss-national-council"
python3 pt_swiss_national_council.py -nt 51 -nm 101 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_snc_naive_coherence_samp_tv_interp"

# BSDS300 #
echo "BSDS300"
python3 pt_bsds.py -sub 'both' -gtype 'grid_and_patches' -nt 15 -nm 25 -sd "naive_tv_coherence" -rf "tv_interpolation" -fn "pt_bsds300_naive_coherence_samp_tv_interp"


## 4. Phase transition: Jump-set coherence sampling and TV recovery ##
echo "4. Phase transitions: jumpt-set coherence sampling and TV interpolation"

# 2-SSBM #
echo "2-SSBM"
python3 pt_ssbm.py -nv 1000 -nc 2 -nt 25 -b 0.5 -na 51 -nm 51 -sd "jump_set_tv_coherence" -rf "tv_interpolation" -fn "pt_2ssbm_jump_set_coherence_samp_tv_interp"

# Unbalanced 2-SBM #
echo "Unbalanced 2-SBM"
python3 pt_sbm.py -nv 1000 -nc 2 -nvpc 200 800 -nt 25 -b 0.5 -na 51 -nm 51 -sd "jump_set_tv_coherence" -rf "tv_interpolation" -fn "pt_2sbm_jump_set_coherence_samp_tv_interp"

# email-EU-core #
echo "email-EU-core"
python3 pt_email_eu_core.py -nc 5 -nt 51 -nm 101 -sd "jump_set_tv_coherence" -rf "tv_interpolation" -fn "pt_email_eu_core_jump_set_coherence_samp_tv_interp"

# swiss-national-council #
echo "swiss-national-council"
python3 pt_swiss_national_council.py -nt 51 -nm 101 -sd "jump_set_tv_coherence" -rf "tv_interpolation" -fn "pt_snc_jump_set_coherence_samp_tv_interp"

# BSDS300 #
echo "BSDS300"
python3 pt_bsds.py -sub 'both' -gtype 'grid_and_patches' -nt 15 -nm 25 -sd "jump_set_tv_coherence" -rf "tv_interpolation" -fn "pt_bsds300_jump_set_coherence_samp_tv_interp"