#!/bin/bash
#SBATCH --partition gpu      # partition (queue)
#SBATCH --nodes 1                # number of nodes
#SBATCH --mem 20G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-04:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output depth_likelihood_clean%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error depth_likelihood_clean%j.err  # filename for STDERR

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
python3 train_depth.py --model laplace --learning-rate 0.00005
date
end=`date +%s`
runtime=$((end-start))

echo Runtime Laplace $runtime
echo Gaussian
python3 train_depth.py --model gaussian --learning-rate 0.00005
date
end=`date +%s`
runtime=$((end-start))

echo Runtime Gaussian $runtime
