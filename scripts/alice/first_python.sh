#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="cavierescarrera@strw.leiden.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=100M
#SBATCH --time=99:00:00
#SBATCH --partition=cpu-medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python/3.11.5-GCCcore-13.2.0
module load mpi4py/3.1.4-gompi-2023a

# Source the Python virtual environment
source $HOME/speedytest/bin/activate

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 /home/cavierescarreramc/HVS_1/scripts/download_gaia_alice.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"