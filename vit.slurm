#!/bin/bash                                                                                                                                                                                                                                                                                               
#SBATCH -A izx@v100
#SBATCH --job-name=sentvit     # job name
#SBATCH -C v100-32g #32g
#SBATCH --qos=qos_gpu-t3 # t3#4 #dev #t4 # QoS
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --mail-type=ALL                     # Email alert types: BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=paul.tresson@cirad.fr
#SBATCH --output=/gpfswork/rech/izx/udr86uu/sentinel/logs/vit_base_patch16_224/%j/logs.out
#SBATCH --error=/gpfswork/rech/izx/udr86uu/sentinel/logs/vit_base_patch16_224/%j/logs.err

### change WORLD_SIZE as gpus/node * num_nodes
export WORLD_SIZE=32

set -x

module purge
module load pytorch-gpu/py3/2.0.0
export GDAL_DATA="/gpfslocalsup/pub/anaconda-py3/2023.03/envs/pytorch-gpu-2.0.0+py3.10.9/lib/python3.10/site-packages/rasterio/gdal_data"
export PROJ_DATA="/gpfslocalsup/pub/anaconda-py3/2023.03/envs/pytorch-gpu-2.0.0+py3.10.9/lib/python3.10/site-packages/rasterio/proj_data"

srun python -u main_dino.py --arch vit_base_patch16_224.dino --batch_size_per_gpu 32 \
--output_dir /gpfswork/rech/izx/udr86uu/sentinel/logs/vit_base_patch16_224/$SLURM_JOB_ID/ \
--mean_dataset 118.61897209826,108.64862578646,74.299567137674,128.58062678415,78.264311585937,122.89514688913,137.14044551509,139.30259245283,72.188204447851,46.75183561633 \
--sd_dataset 53.144739238236,49.092660967468,42.331142946847,47.635403548161,44.975424397381,47.834278860924,45.405740935281,45.354688621251,45.220474931332,41.992920448203 
