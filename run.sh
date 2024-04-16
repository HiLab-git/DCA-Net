#! /usr/bin/env sh

srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 -c 6 --job-name=python --kill-on-bad-exit=1 python train_multi_fundus.py
