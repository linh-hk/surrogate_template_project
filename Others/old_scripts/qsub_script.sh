#$ -l tmem=80G
#$ -l h_vmem=80G
#$ -l h_rt=36:00:00 

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N Simulation_MPI_run

#$ -cwd

#The code you want to run now goes here.

python3 -u execute_correlation_surrogate_tests.py caroline_LvCh_FitzHugh_100 100
