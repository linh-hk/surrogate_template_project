#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=15:00:00
#$ -pe smp 16
#$ -R y

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N "multispecies"

#$ -cwd

#The code you want to run now goes here.
hostname
date
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 -u Simulation_code/run_test/execute.py $1 $2 $3 $4 $5

