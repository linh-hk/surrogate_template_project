#$ -l tmem=1G
#$ -l h_vmem=1G
#$ -l h_rt=00:05:00 

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N test

#$ -cwd

#The code you want to run now goes here.

python3 -u test.py $1
