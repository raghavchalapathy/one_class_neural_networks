#!/usr/bin/env bash

mkdir ../log/mnist;
mkdir ../log/mnist/deepSVDD;

for exp in $(seq 0 9);
  do
    mkdir ../log/mnist/deepSVDD/${exp}vsall;
  done

sh experiments/mnist_svdd_exp_seeds.sh 0 &

wait
echo all experiments complete
