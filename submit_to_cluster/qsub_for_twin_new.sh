#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=35:00:00
#$ -pe smp 8
#$ -R y

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N "LV_extra_twin"

#$ -cwd

#The code you want to run now goes here.
hostname
date
python3 -u Simulation_code/run_test/execute_LVextra_twin2.py $1
