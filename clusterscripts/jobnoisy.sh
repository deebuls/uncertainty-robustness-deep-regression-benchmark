#!/bin/bash
#SBATCH --partition gpu      # partition (queue)
#SBATCH --nodes 1                # number of nodes
#SBATCH --ntasks-per-node=32    #number of cpu
#SBATCH --mem 80G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-04:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output noisy_depth%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error noisy_depth%j.err  # filename for STDERR

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
start=`date +%s`
#python /home/dnair2m/evidential_regression/run_cubic_tests.py
python train_depth.py --model evidential --learning-rate 0.00005 --noisy-data
date
end=`date +%s`
runtime=$((end-start))

echo Evidential $runtime
printf '%dh:%dm:%ds\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
date
echo Starting laplace
start=`date +%s`
python3 train_depth.py --model laplace --learning-rate 0.00005 --noisy-data
date
end=`date +%s`
runtime=$((end-start))

echo Runtime Laplace $runtime
printf '%dh:%dm:%ds\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
echo Starting Gaussian
start=`date +%s`
python3 train_depth.py --model gaussian --learning-rate 0.00005 --noisy-data
date
end=`date +%s`
runtime=$((end-start))

echo Runtime Gaussian $runtime
printf '%dh:%dm:%ds\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
