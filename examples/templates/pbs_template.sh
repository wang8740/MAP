#!/bin/bash
#PBS -N alignmap_job
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=16gb
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

# Load environment modules if needed
# module load python/3.8

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Starting time: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Commands to execute
COMMAND_PLACEHOLDER

echo "Finished at: $(date)" 