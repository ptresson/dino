#!/bin/bash

#SBATCH --job-name=test-run1
#SBATCH --account=udr86uu
#SBATCH -A izx@v100
#SBATCH -C v100-16g
#SBATCH --qos=qos_gpu-t3                    # Subscribe to a Quality of service
#SBATCH --time=3:00:00                     # Time limit; either min or ddhh:mm:ss
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=4                 # Processes per node
#SBATCH --cpus-per-task=10                  # CPU cores per process
#SBATCH --gres=gpu:4                        # GPUs per nodes
#SBATCH --ntasks=4
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --mail-type=ALL                     # Email alert types: BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=paul.tresson@cirad.fr
#SBATCH --output=/gpfswork/rech/izx/udr86uu/dino/logs/%j.out
#SBATCH --error=/gpfswork/rech/izx/udr86uu/dino/logs/%j.err

# Module
module purge
module load pytorch-gpu/py3/2.0.0
conda activate torchgeo

# Print command that are executed
export PYTHONUNBUFFERED=1;
set -x

# Executing python code
export PYTHONPATH=":."
python -u main_dino.py
