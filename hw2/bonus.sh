#!/bin/bash
echo "arg:$1"
echo "arg:$2"
echo "arg:$3"
echo "arg:$4"

# lambda = 1
# echo "python train_pg_f18.py Walker2d-v2 -ep 150 --discount $3 -n 35 -e 1 --seed 1 -l 2 -s 32 -b $1 -lr $2 -rtg --nn_baseline --exp_name wa_b$1_r$2_gamma$3_gae1"
# python train_pg_f18.py Walker2d-v2 -ep 150 --discount $3 -n 35 -e 1 --seed 1 -l 2 -s 32 -b $1 -lr $2 -rtg --nn_baseline --exp_name wa_b$1_r$2_gamma$3_gae1 > /dev/null &

echo "python train_pg_f18.py Walker2d-v2 -ep 150 --discount $3 -n 35 -e 1 --seed 1 -l 2 -s 32 -b $1 -lr $2 -rtg --nn_baseline --gae --gae_lambda $4 --exp_name  wa_b$1_r$2_gamma$3_lambda$4"
# python train_pg_f18.py Walker2d-v2 -ep 150 --discount $3 -n 35 -e 1 --seed 1 -l 2 -s 32 -b $1 -lr $2 -rtg --nn_baseline --gae --gae_lambda $4 --exp_name  wa_b$1_r$2_gamma$3_lambda$4 > /dev/null &