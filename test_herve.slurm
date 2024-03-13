#!/bin/bash                                                                                                                                                                                                                                                                                               
#SBATCH -A izx@v100
#SBATCH --job-name=dinosent     # job name
#SBATCH -C v100-32g #32g
#SBATCH --qos=qos_gpu-dev # t3#4 #dev #t4 # QoS
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=02:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --mail-type=ALL                     # Email alert types: BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=paul.tresson@cirad.fr
#SBATCH --output=/gpfswork/rech/izx/udr86uu/sentinel/logs/%j.out
#SBATCH --error=/gpfswork/rech/izx/udr86uu/sentinel/logs/%j.err

### change WORLD_SIZE as gpus/node * num_nodes
export WORLD_SIZE=8

set -x

module purge
module load pytorch-gpu/py3/2.0.0
export GDAL_DATA="/gpfslocalsup/pub/anaconda-py3/2023.03/envs/pytorch-gpu-2.0.0+py3.10.9/lib/python3.10/site-packages/rasterio/gdal_data"
export PROJ_DATA="/gpfslocalsup/pub/anaconda-py3/2023.03/envs/pytorch-gpu-2.0.0+py3.10.9/lib/python3.10/site-packages/rasterio/proj_data"

#srun python -u run_with_submitit.py
srun python -u main_dino.py --corrected False --arch vit_base_patch16_224 --batch_size_per_gpu 32

