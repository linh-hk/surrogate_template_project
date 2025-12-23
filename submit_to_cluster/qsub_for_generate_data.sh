#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=03:00:00
#$ -pe smp 8
#$ -R y

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N "generate_data_multispecies_symcomp"

#$ -cwd

#The code you want to run now goes here.
hostname
date
python3 -u Simulation_code/run_test/execute_generate_data.py