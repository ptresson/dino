#!/bin/bash

#SBATCH --job-name=gpn-run1
#SBATCH --account=ugq71iu
#SBATCH -A igz@v100
#SBATCH -C v100-16g
#SBATCH --qos=qos_gpu-t3                    # Subscribe to a Quality of service
#SBATCH --time=3:00:00                     # Time limit; either min or ddhh:mm:ss
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=4                 # Processes per node
#SBATCH --cpus-per-task=12                  # CPU cores per process
#SBATCH --gres=gpu:4                        # GPUs per nodes
#SBATCH --mail-type=ALL                     # Email alert types: BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=remi.palard@cirad.fr
#SBATCH --output=/gpfswork/rech/igz/ugq71iu/geoplantnet/logs/%j.out
#SBATCH --error=/gpfswork/rech/igz/ugq71iu/geoplantnet/logs/%j.err

# Module
module purge
module load intel-compilers/19.0.4
module load pytorch-gpu/py3/2.0.0

# Print command that are executed
export PYTHONUNBUFFERED=1;
set -x

# Executing python code
export PYTHONPATH=":."
export GDAL_DATA="/gpfslocalsup/pub/anaconda-py3/2023.03/envs/pytorch-gpu-2.0.0+py3.10.9/lib/python3.10/site-packages/rasterio/gdal_data"
export PROJ_DATA="/gpfslocalsup/pub/anaconda-py3/2023.03/envs/pytorch-gpu-2.0.0+py3.10.9/lib/python3.10/site-packages/rasterio/proj_data"
python main.py --workers 12
