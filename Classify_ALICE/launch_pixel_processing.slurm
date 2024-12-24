#!/bin/bash
#SBATCH --job-name=pixel_processing_test
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="cavierescarrera@strw.leiden.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=64000M               # 64 GB of memory for all tasks
#SBATCH --partition="gpu_strw"       # Specify the partition
#SBATCH --account="gpu_strw"
#SBATCH --time=7-0:00:00           # Maximum runtime
#SBATCH --ntasks=10                 # 8 MPI tasks
#SBATCH --cpus-per-task=1          # 1 CPU per task (for pure MPI jobs)

# Load modules
module load Python/3.11.5-GCCcore-13.2.0
module load mpi4py/3.1.4-gompi-2023a

# Activate the Python virtual environment
source $HOME/HVS_1/enviroments/speedytest/bin/activate

# Job information
echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
export CWD=$(pwd)
echo "[$SHELL] ## Current working directory: $CWD"

# Run the Python script with MPI
echo "[$SHELL] ## Run script"
srun python pixel_processing_MPI.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"
