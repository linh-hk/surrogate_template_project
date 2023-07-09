#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=72:00:00 

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N r_59

#$ -cwd

#The code you want to run now goes here.

python3 -u execute_for_generated_data_change_r.py $1
