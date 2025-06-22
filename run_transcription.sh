#!/bin/bash
export USE_NNPACK=0
export OMP_NUM_THREADS=4
python quick_test_runner.py "$@"
