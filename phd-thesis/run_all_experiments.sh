#!/bin/bash

echo "Running all experiments. This will take a long time. You could also try running them line by line."

## 1. Phase transition: Uniform sampling and TV recovery ##
sh run_experiment_1.sh

## 2. Phase transition: Uniform sampling and Dirichlet recovery ##
sh run_experiment_2.sh

## 3. Phase transition: Naive coherence sampling and TV recovery ##
sh run_experiment_3.sh

## 4. Phase transition: Jump-set coherence sampling and TV recovery ##
sh run_experiment_4.sh