# surrogate_template_project
This project is for my MRes degree at UCL. It contains code for simulation of several models, the surrogate test for dependence between time series and their execution scripts on UCL cluster. 

The scripts are organised into 4 main sectors:
1. surrogate_dependence_test: where I put the codes to generate the data and the codes for surrogate tests (functions of correlation statistics and functions of surrogate protocols and wrappers of the two)
2. run_test: where I put scripts for executions of the functions in the surrogate_dependence_test. It specifies which model I simulated and which surrogate tests I ran.
3. submit_to_cluster: where I put the scripts to submit the scripts in 'run_test' to the UCL cluster.
4. after_cluster: where I use the results from the cluster to generate figures.

## Installation and execution:
  Step1: Clone this repository to cluster or where you want to generate data and run surrogate tests:
  
    Current working directory: $ PROJPATH=$(pwd)
    Clone the git repos under the name Simulation_code which means that the cloned repos will be at ${PROJPATH}/surrogate_template_project and you will rename the "surrogate_template_project" into "Simulation_code" or directly with git clone:
    
    `git clone https://github.com/linh-hk/surrogate_template_project/tree/main Simulation_code`
    
  Step2: Adjust the scripts to make sure the directories are correct:
  
    Any scripts that use sys.path.append(), change:
      + "/home/hoanlinh/Simulation_test/" to your current working directory - ${PROJPATH} (don't literally write ${PROJPATH}, write the actual absolute path) 
    (the scripts include surrogate_dependence_test.main.py, surrogate_dependence_test.ccm_xory.py, all scripts in run_test)
    Any scripts that use os.chdir(), change:
      + "C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/" to your current working directory
      
  Step3: Generate your data so that the path to generated data is at
  
    ${PROJPATH}/xy_*/data.pkl or ${PROJPATH}/*500*/data.pkl (where * stands for any characters according to regular expressions)
    
    (I apologise for this inconvenience but this is how the program is currently written ...)
    
    Do this by 
    
    $ qsub Simulation_code/submit_to_cluster/qsub_for_generate_data.sh (run this at your current working directory)
  
  Step4: Run surrogate test
  
    Example:
    `$ qsub Simulation_code/submit_to_cluster/qsub_for_twin_new.sh xy_* 100` (the reason there is a 100 or any number from 0 to 900 is that for xy_* folders, I generated 1000 trials but I only want to run trials from the 100th to the 100+100 th)
    `$ qsub Simulation_code/submit_to_cluster/qsub_for_twin_LVextra.sh *500*`
    `$ qsub Simulation_code/submit_to_cluster/qsub_for_normalised_data.sh twin * 100` (for the recently added normalising scripts I generalised twin, tts, randphase into one type of batch, * is any model name, and 100 or any number from 0 to 900 is optional)

## surrogate_dependence_test:
+ Scripts for generating models are in GenerateData.py
+ Scripts for correlation statistics are in correlation_stats.py, ccm_xory.py and granger_xory.py
+ Scripts for surrogate protocols and wrapper for surrogate tests are in main.py
+ Scripts for multiprocessing that allow for parallelising jobs of a parallelised job are in multiprocessor.py (functions that use this module include ccm_xory.ccm_predict_surr(), ccm_xory.ccm_surr_predict(), main.scan_lags(),

## run_test:
+ Scripts to use GenerateData.py with varying parameters are in execute_generate_data.py
+ I have scripts to run 3 types of batches, the LVextra, the 'new_Caroline', the 'new' and the SCOR_DOT. Recently there is the normalising which includes loading data into the cluster environment, normalising the data and then executing the scripts in 'main'.
+++ For each types, I have scripts for true positive (no tags) and for false positive (_tagged falsepos).
+++ Scripts of the 'LVextra' and 'new' are also tagged with the name of the surrogate protocols (randphase, tts or twin) (they could have been written in one script but I find it more convenient to keep track if separate them so I left them separated, it's the cluster's thing)

## submit_to_cluster:
+ Similar to run_test, I have qsub scripts for each jobs and each types of batches. As mentioned earlier with the 'randphase', 'tts', 'twin' tags, inside these scripts there is the argument -N where each job on the cluster can be named and traced, I have qsub scripts for each surrogate protocol separately.
+ These qsub scripts are submitted as mentioned earlier in this README.md

## after_cluster:
+ Scripts in the after_cluster are specifically run on local computers that support graphics.
+ I either download the results from the cluster through WinSCP on Windows or File on Ubuntu. Then, I execute the scripts.

## Others:
+ All types of draft scripts and scripts that Alex and Caroline sent me to start out the project.
