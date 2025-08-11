#!/bin/bash
#SBATCH --job-name="UNET_L1_loss"
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00  

# Directories for logs (no error directory needed if using the same log file)
log_dir="/home/woody/iwi5/iwi5240h/Masters-Thesis/log"
mkdir -p "$log_dir"

# Log file path
log_file="$log_dir/$SLURM_JOB_NAME"_"$SLURM_JOB_ID.log"

# Redirect both stdout and stderr to the same log file
exec >"$log_file" 2>&1

# Display job details in the log file
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on Host: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Run the Python model
python main.py
