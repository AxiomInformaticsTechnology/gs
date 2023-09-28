#!/bin/bash

echo 'Local Gibbs Sampler Experiment'

modes=(serial mpi cuda cuda-mpi)
persons=(500 1000 2000 4000 6000 8000 10000 15000 20000 25000 50000)
items=(20 50 100 200)
nodes=(5 10)

mkdir -p "output"

for m in "${modes[@]}"
do
  mkdir -p "output/$m"
  for p in "${persons[@]}"
  do
    for i in "${items[@]}"
    do
      if [[ $m == "mpi" ]] || [[ $m == "cuda-mpi" ]]
      then
        for n in "${nodes[@]}"
        do
          echo "mpiexec -n $n uno2p $m $p $i"
          mpiexec -n $n uno2p $m $p $i > "output/$m/$m.$n.$p.$i.out"
        done
      else
        echo "uno2p $m $p $i"
        uno2p $m $p $i  > "output/$m/$m.$p.$i.out"
      fi
    done
  done
done
