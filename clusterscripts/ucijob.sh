#!/bin/bash
#SBATCH --nodes 1                # number of nodes
#SBATCH --time 1-04:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output uci_gauss%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error uci_gauss%j.err  # filename for STDERR

module load cuda
module load gcc/9.1.0

source /home/dnair2m/miniconda3/bin/activate
conda activate evidence
python --version
# change to submit directory (with executable)
cd $PBS_O_WORKDIR
# execute sequential program
which python
cd /home/dnair2m/evidential_regression
pwd
date
echo laplace
start=`date +%s`
python3 run_uci_dataset_tests.py --datasets yacht boston concrete energy-efficiency kin8nm naval power-plant protein wine --num-trials 20
date
end=`date +%s`
runtime=$((end-start))
printf '%dh:%dm:%ds\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
