#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=06:00:00
#$ -pe smp 8
#$ -R y

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N "multispecies_s2_nolag99"

#$ -cwd

#The code you want to run now goes here.
hostname
date
python3 -u Simulation_code/run_test/execute.py $1 $2 $3 $4 $5

