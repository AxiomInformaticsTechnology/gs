#!/bin/bash

nre='^[0-9]+$'

if [[ $# < 4 ]] || ! [[ $1 =~ $nre ]] || [[ $1 < 1 ]] || ([[ "$2" == *"mpi"* ]] && [[ $# < 6 ]])
then
  echo "midway.sh [iterations] [modes] [persons] [items] <nodes> <tasks>"
  echo "iterations: n >= 1"
  echo "modes: serial,mpi,cuda,cuda-mpi"
  echo "persons: 500,1000,2000,..."
  echo "items: 20,50,..."
  echo "nodes: 1,2,..."
  echo "tasks: 1,5,10,..."
  echo ""
  echo "if mpi mode nodes and tasks required"
  exit 1
fi

iterations=$1
modes=($(echo $2 | tr "," "\n"))
persons=($(echo $3 | tr "," "\n"))
items=($(echo $4 | tr "," "\n"))

echo 'Midway 3 Gibbs Sampler Experiment'
echo "iterations: $1"
echo "modes: $2"
echo "persons: $3"
echo "items: $4"

if [[ "$2" == *"mpi"* ]]
then
  nodes=($(echo $5 | tr "," "\n"))
  tasks=($(echo $6 | tr "," "\n"))

  echo "nodes: $5"
  echo "tasks: $6"
fi

mkdir -p "/home/wwelling/gs/sbatch"

for m in "${modes[@]}"
do
  mkdir -p "/home/wwelling/gs/sbatch/$m"
  mkdir -p "/home/wwelling/gs/sbatch/$m/output"
  mkdir -p "/home/wwelling/gs/sbatch/$m/error"
  for p in "${persons[@]}"
  do
    for i in "${items[@]}"
    do
      if [[ $m == "mpi" ]] || [[ $m == "cuda-mpi" ]]
      then
        for n in "${nodes[@]}"
        do
          for t in "${tasks[@]}"
          do
            for ((r = 0; r < $iterations; r++));
            do
              np=$(($n * $t))
              gpus=$t
              if [ "$t" -gt 4 ]
              then
                gpus=4
              fi
              (
                echo "#!/bin/bash"
                echo "#SBATCH --job-name=uno2p.$m.$n.$t.$p.$i.$r"
                echo "#SBATCH --account=pi-yysheng"
                echo "#SBATCH --output=/home/wwelling/gs/sbatch/$m/output/uno2p.$m.$n.$t.$p.$i.$r.out"
                echo "#SBATCH --error=/home/wwelling/gs/sbatch/$m/error/uno2p.$m.$n.$t.$p.$i.$r.err"
                echo "#SBATCH --time=72:00:00"
                echo "#SBATCH --partition=gpu"
                if [[ $m == "cuda-mpi" ]]
                then
                  echo "#SBATCH --mem-per-cpu=16GB"
                  echo "#SBATCH --gres=gpu:$gpus"
                  echo "#SBATCH --constraint=v100"
                else
                  echo "#SBATCH --mem=64G"
                fi
                echo "#SBATCH --nodes=$n"
                echo "#SBATCH --ntasks-per-node=$t"
                echo "#SBATCH --cpus-per-task=1"

                echo "module load openmpi-cuda-aware"

                echo "mpirun -np $np /home/wwelling/gs/bin/uno2p $m $p $i"
              ) > "/home/wwelling/gs/sbatch/$m/uno2p.$m.$n.$t.$p.$i.$r.sbatch"
              echo "/home/wwelling/gs/sbatch/$m/uno2p.$m.$n.$t.$p.$i.$r.sbatch"
              sbatch "/home/wwelling/gs/sbatch/$m/uno2p.$m.$n.$t.$p.$i.$r.sbatch"
            done
          done
        done
      else
        for ((r = 0; r < $iterations; r++));
        do
          (
            echo "#!/bin/bash"
            echo "#SBATCH --job-name=uno2p.$m.$p.$i.$r"
            echo "#SBATCH --account=pi-yysheng"
            echo "#SBATCH --output=/home/wwelling/gs/sbatch/$m/output/uno2p.$m.$p.$i.$r.out"
            echo "#SBATCH --error=/home/wwelling/gs/sbatch/$m/error/uno2p.$m.$p.$i.$r.err"
            echo "#SBATCH --time=72:00:00"
            echo "#SBATCH --partition=gpu"
            if [[ $m == "cuda" ]]
            then
              echo "#SBATCH --mem-per-cpu=16GB"
              echo "#SBATCH --gres=gpu:1"
              echo "#SBATCH --constraint=v100"
            else
              echo "#SBATCH --mem=64G"
            fi
            echo "#SBATCH --nodes=1"
            echo "#SBATCH --ntasks-per-node=1"
            echo "#SBATCH --cpus-per-task=1"

            echo "module load openmpi-cuda-aware"

            echo "/home/wwelling/gs/bin/uno2p $m $p $i"
          ) > "/home/wwelling/gs/sbatch/$m/uno2p.$m.$p.$i.$r.sbatch"
          echo "/home/wwelling/gs/sbatch/$m/uno2p.$m.$p.$i.$r.sbatch"
          sbatch "/home/wwelling/gs/sbatch/$m/uno2p.$m.$p.$i.$r.sbatch"
        done
      fi
    done
  done
done
