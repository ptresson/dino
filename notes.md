# Environement

installation locale de torchgeo et submitit via pip

```
[udr86uu@jean-zay4: logs]$ export PYTHONUSERBASE=$WORK/.local/
[udr86uu@jean-zay4: logs]$ pip install --user --no-cache-dir torchgeo
[udr86uu@jean-zay4: logs]$ pip install --user --no-cache-dir submitit
```

problème avec PROJ, ajout des lignes
```
export GDAL_DATA="/gpfslocalsup/pub/anaconda-py3/2023.03/envs/pytorch-gpu-2.0.0+py3.10.9/lib/python3.10/site-packages/rasterio/gdal_data"
export PROJ_DATA="/gpfslocalsup/pub/anaconda-py3/2023.03/envs/pytorch-gpu-2.0.0+py3.10.9/lib/python3.10/site-packages/rasterio/proj_data"
```
dans le script slurm

# Modifs code
dans `run_with_submitit.py`

commenter le contenu de  `get_shared_folder()` pour que ça retourne juste `args.output_dir`
