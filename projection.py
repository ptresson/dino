# general
import warnings
import joblib
import os
import sys
from torch import nn
from collections import OrderedDict

# pytorch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import kornia.augmentation as K
from kornia.constants import Resample
from kornia.enhance.normalize import Normalize
import timm
from timm import create_model

# torchgeo
from torchgeo.datasets import RasterDataset, BoundingBox, stack_samples, unbind_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential

# geo
import geopandas as gpd
import rasterio

# data
import numpy as np
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.cluster import KMeans
import hdbscan
warnings.filterwarnings("ignore")

from main_dino import vit_first_layer_with_nchan, resnet_first_layer_with_nchan
from main_dino import MergedDataLoader
# custom modules
# from utils import change_tif_resolution
# from utils import vit_first_layer_with_nchan, reconstruct_img
# from utils import numpy_to_geotif
# from utils import export_on_map
# from utils import get_mean_sd_by_band, get_crs
# from utils import remove_black_tiles, remove_empty_tiles, remove_black_tiles2
# from custom_datasets import *


def fit_proj(
        model,
        cls_token=False,
        feat_dim=1024,
        batch_size=10,
        exclude_value=-77.04,
        nsamples=1_000,
        size=224,
        patch_h=100,
        patch_w=100,
        method='pca',
        out_path_proj_model="out/proj.pkl",
        roi=None,
        n_components=3,
        n_neighbors=10,
        min_samples=5,
        min_cluster_size=100,
        n_clusters=8,
        ):
    """
    Fit a projection or clustering algorithm in the feature space of a deep learning model.

    Args:
        dataset (RasterDataset): Torchgeo raster dataset containing the data.
        model: PyTorch model to use for projection into the feature space.
        cls_token (bool, optional): ViT specific. Is the projection fitted at image (cls_token) or patch (patch_token) level. `cls_token=False` will default to use at the patch token level.
            (default=False).
        feat_dim (int, optional): Dimension of the feature space output by the backbone. This value depends on the chosen architecture.
            (default=1024).
        batch_size (int, optional): how many samples per batch to load
            (default: 10).
        exclude_value (float, optional): If a value is provided, tiles whose cumulated sum is below that value will be removed.
            (default: -77.04)
        nsamples (int, optional): The number of samples to fit the projection or clustering algorithm.
            (default: 100)
        size (int, optional): The size of the sampled images (in pixels).
            (default: 224)
        patch_h (int, optional): The height of the patches used for analysis. It depends on the desired spatial resolution.
            (default: 100)
        patch_w (int, optional): The width of the patches used for analysis. It depends on the desired spatial resolution.
            (default: 100)
        method (str, optional): The method used for fitting the projection or clustering. Choose between 'pca', 'umap', 'hdbscan', and 'kmeans'.
            (default: 'pca')
        out_path_proj_model (str, optional): The path to save the fitted projection or clustering model to a .pkl file.
            (default: "out/proj.pkl")
        roi (tuple, optional): The region of interest (ROI) for the analysis. By default, it takes the entire dataset bounds.
            (default: None)
        n_components (int, optional): The number of components for PCA or UMAP.
            (default: 3)
        n_neighbors (int, optional): The number of neighbors for UMAP.
            (default: 10)
        min_samples (int, optional): The minimum number of samples for HDBSCAN.
            (default: 5)
        min_cluster_size (int, optional): The minimum cluster size for HDBSCAN.
            (default: 100)
        n_clusters (int, optional): The number of clusters for KMeans.
            (default: 8)
    """

    # # If ROI is smaller than the full dataset
    # if roi:
    #     sampler = RandomGeoSampler(dataset, size=size, length=nsamples, roi=roi)
    # else:
    #     sampler = RandomGeoSampler(dataset, size=size, length=nsamples)

    # dataloader = DataLoader(
    #     dataset, 
    #     sampler=sampler, 
    #     collate_fn=stack_samples, 
    #     shuffle=False, 
    #     batch_size=batch_size
    # )
    dataloader = prepare_congo_data_proj(size=size, nsamples=nsamples)

    feat_img = None  # Initialize the feature tensor

    i = 0
    for batch in dataloader:
        print(f"{i / len(dataloader):.2%}", end="\r")

        images = batch['image']
        if len(images.shape) > 4:
            images = images.squeeze(1)
        bboxes = batch['bbox']
        images = images.type(torch.float)
        images = images.cuda()

        do_forward = True

        # if the tensor contains the exclue value, it is not processed.
        # the purpose is to handle black or patially empty tiles
        if exclude_value:
            images, _, is_non_empty = remove_black_tiles2(images, bboxes, exclude_value)
            if not is_non_empty:
                do_forward = False

        if do_forward:
            with torch.no_grad():
                if hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features')):
                    features_dict = model.forward_features(images)

                    if cls_token:
                        try:
                            feat = features_dict['x_norm_clstoken']
                        except (KeyError, IndexError):
                            feat = features_dict
                        feat = feat.detach().cpu().numpy()
                        if feat_img is None:
                            feat_img = feat
                        feat_img = np.concatenate((feat_img, feat), axis=0)
                    else:

                        try:
                            feat = features_dict['x_norm_patchtokens']
                        except KeyError:
                            raise NotImplementedError("model architecture does not support patch_tokens inference")

                        feat = feat.detach().cpu().numpy()
                        feat = feat.reshape(images.shape[0] * patch_h * patch_w, feat_dim)
                        if feat_img is None:
                            feat_img = feat
                        feat_img = np.concatenate((feat_img, feat), axis=0)
                else:
                    feat = model(images)
                    if feat_img is None:
                        feat_img = feat
                    feat_img = np.concatenate((feat_img, feat), axis=0)

        i += 1

    # Using different methods to fit projection/clustering on the feature space.
    # Each method comes with its own hyperparameters.
    if method == 'pca':
        embedder = PCA(n_components=n_components)
    elif method == 'umap':
        embedder = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    elif method == 'hdbscan':
        embedder = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            prediction_data=True
        )
    elif method == 'kmeans':
        embedder = KMeans(n_clusters=n_clusters)

    embedder = embedder.fit(feat_img)

    # Saving to a .pkl file
    joblib.dump(embedder, out_path_proj_model)


def inf_proj(
        dataset,
        model,
        cls_token=False,
        feat_dim=1024,
        out_dim=3,
        batch_size=10,
        size=224,
        patch_h=100,
        patch_w=100,
        path_proj_model="out/proj.pkl",
        roi=None,
        path_out=None,
        ):
    """
    Perform inference using a pre-fitted projection model on the feature space of a deep learning model.

    Args:
        dataset (RasterDataset): Torchgeo raster dataset containing the data.
        model: PyTorch model to use for feature extraction and inference.
        cls_token (bool, optional): ViT specific. If True, works at cls_token (image) scale; otherwise, works at patch scale.
            (default=False).
        out_dim (int, optional): Dimension of the output feature space after projection.
            (default: 3)
        feat_dim (int, optional): Dimension of the feature space output by the backbone. This value depends on the chosen architecture.
            (default=1024).
        batch_size (int, optional): How many samples per batch to load.
            (default: 10).
        size (int, optional): The size of the tiles fed to the model (in pixels).
            (default: 224)
        patch_h (int, optional): The height of the patches used for analysis. It depends on the desired spatial resolution.
            (default: 100)
        patch_w (int, optional): The width of the patches used for analysis. It depends on the desired spatial resolution.
            (default: 100)
        path_proj_model (str, optional): The path to the pre-fitted projection model stored in a .pkl file.
            (default: "out/proj.pkl")
        roi (tuple, optional): The region of interest (ROI) for the analysis. By default, it takes the entire dataset bounds.
            (default: None)
        path_out (str, optional): The path to save the output feature space. If None, the output is not saved.
            (default: None)

    Returns:
        tuple: A tuple containing the macro image of the feature space and the bounding boxes of the processed samples.
    """

    # Load pre-fitted projection or clustering model
    embedder = joblib.load(path_proj_model)

    # Set up data loader based on the specified region of interest (ROI)
    if roi:
        sampler = GridGeoSampler(dataset, size=size, stride=size, roi=roi)
    else:
        sampler = GridGeoSampler(dataset, size=size, stride=size)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=stack_samples,
        shuffle=False,
        batch_size=batch_size
    )

    N = len(dataloader)
    bboxes = []  # List to store bounding boxes of processed samples

    # Initialize tensor used to store the projected feature space
    feat_img = None

    i = 0
    for batch in dataloader:
        print(f"{i / N:.2%}", end="\r")

        images = batch['image']
        if len(images.shape) > 4:
            images = images.squeeze(1)
        images = images.type(torch.float)
        images = images.cuda()

        # Extract bounding boxes for each sample
        for sample in unbind_samples(batch):
            bboxes.append(sample['bbox'])

        with torch.no_grad():
            features_dict = model.forward_features(images)

            if cls_token:
                feat = features_dict['x_norm_clstoken']
                feat = feat.detach().cpu().numpy()
                feat = feat.reshape(images.shape[0], feat_dim)
            else:
                feat = features_dict['x_norm_patchtokens']
                feat = feat.detach().cpu().numpy()
                feat = feat.reshape(images.shape[0] * patch_h * patch_w, feat_dim)

            # Transform features using the pre-fitted embedder model
            red_features = embedder.transform(feat)

            # Reshape the transformed features to match the desired output shape
            if cls_token:
                red_features = red_features.reshape(images.shape[0], out_dim)
            else:
                red_features = red_features.reshape(images.shape[0], patch_h, patch_w, out_dim)

            # Concatenate the transformed features to the existing tensor
            if feat_img is None:
                feat_img = red_features
            else:
                feat_img = np.concatenate((feat_img, red_features), axis=0)

        i += 1

    # Determine the grid dimensions based on bounding boxes
    Nx = len(bboxes)
    for i in range(1, len(bboxes)):
        if bboxes[i][0] < bboxes[i - 1][0]:
            Nx = i
            break
    Ny = int(len(bboxes) / Nx)

    # Generate a macro image from the projected feature space
    macro_img = reconstruct_img(feat_img, Nx, Ny)

    # Save the macro image if the output path is specified
    if path_out:
        np.save(path_out, macro_img)

    return macro_img, bboxes




def inf_cluster(
        dataset,
        model,
        batch_size = 10, 
        exclude_value=None,
        size=224, # size of sampled images
        path_proj_model= "out/proj.pkl",
        roi=None, # defaults takes all dataset bounds
        ):
    """
    This function may be depreciated in the sense that performing a clustering on a reprojected geotiff
    will be more reliable and simple.
    Moreover, it only works at cls_token level. First doing a projection and then cluster allows to works
    at patch_token level.
    """

    clusterer = joblib.load(path_proj_model)

    # if roi is samller than full dataset
    if roi:
        sampler = GridGeoSampler(dataset, size = size, stride=size, roi = roi)
    else:
        sampler = GridGeoSampler(dataset, size = size, stride=size)

    dataloader = DataLoader(
            dataset, 
            sampler=sampler, 
            collate_fn=stack_samples, 
            shuffle=False, 
            batch_size = batch_size
            )

    N = len(dataloader)

    bboxes = []
    labels = []

    i=0
    for batch in dataloader:
        print(f"{i/N:.2%}", end="\r")

        images = batch['image']
        if len(images.shape) > 4: # transforms can create a false dim ?
            images = images.squeeze(1)
        batch_bboxes = batch['bbox']
        images = images.type(torch.float)
        images = images.cuda()

        do_forward = True
        if exclude_value:
            images, batch_bboxes, is_non_empty = remove_black_tiles2(images, batch_bboxes, exclude_value)
            if is_non_empty == False:
                do_forward = False
        
        if do_forward:

            bboxes.extend(batch_bboxes)

            with torch.no_grad():

                features_dict = model.forward_features(images)
                try:
                    feat = features_dict['x_norm_clstoken']
                except:
                    feat = features_dict
                feat = feat.detach().cpu().numpy()

                try :
                    label = clusterer.predict(feat)
                # hdbscan prediction do not work the same way as classic kmeans
                except:
                    label, strengths = hdbscan.prediction.approximate_predict(
                                clusterer,
                                feat
                                )
                    label = list(label)

                label = label.astype('int8')
                labels.extend(label)

                del feat
                del features_dict
                del batch_bboxes
        i+=1

    return labels, bboxes

def prepare_congo_data_proj(
        data_path='/home/ptresson/congo/panchro_congo_all_renamed', 
        means=[558.03], 
        sds=[89.63], 
        size=224, 
        nsamples=500,
        batch_size=50,
        ):

    transform = AugmentationSequential(
            K.Resize(size, resample=Resample.BICUBIC.name),
            Normalize(means, sds),
            data_keys=["image"],
    )

    class Raster(RasterDataset):
        filename_glob = '*.tif'
        is_image = True

    dataset1 = Raster(os.path.join(data_path,'A'))
    dataset2 = Raster(os.path.join(data_path,'B'))

    dataset1.transforms = transform
    dataset2.transforms = transform

    bb1 = dataset1.index.bounds
    bb2 = dataset2.index.bounds
    roi1 = BoundingBox(bb1[0], bb1[1], bb1[2], bb1[3], bb1[4], bb1[5])
    roi2 = BoundingBox(bb2[0], bb2[1], bb2[2], bb2[3], bb2[4], bb2[5])

    sampler1 = RandomGeoSampler(
            dataset1, 
            size=(size,size), 
            length=nsamples, 
            roi=roi1
            )
    sampler2 = RandomGeoSampler(
            dataset2, 
            size=(size,size), 
            length=nsamples, 
            roi=roi2
            )

    data_loader1 = DataLoader(
            dataset1, 
            sampler=sampler1, 
            collate_fn=stack_samples, 
            shuffle=False, 
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
            )
    data_loader2 = DataLoader(
            dataset2, 
            sampler=sampler2, 
            collate_fn=stack_samples, 
            shuffle=False, 
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
            )

    data_loader=MergedDataLoader(data_loader1, data_loader2)

    return data_loader

def create_template_model(arch, in_chans, patch_size=16):

    if arch in torch.hub.list("facebookresearch/dino:main"):
        if 'vit' in arch :
            model = torch.hub.load('facebookresearch/dino:main', arch,
                                    in_chans=in_chans,
                                    strict=False,
                                    pretrained=False,
                                    )

        if 'resnet' in arch :

            model = torch.hub.load('facebookresearch/dino:main', arch)
            model = resnet_first_layer_with_nchan(model, in_chans)

        model.fc, model.head = nn.Identity(), nn.Identity()
        # get embed_dim before fully loading model to avoid hardcoding value
        if not hasattr(model, 'embed_dim'):
            x = model(torch.rand(1,in_chans,224,224))
            embed_dim = x.shape[1]
        else:
            embed_dim = model.embed_dim


    elif arch in timm.list_models(pretrained=True):
        model = create_model(
                arch,
                pretrained=True,
                in_chans=in_chans,
                )
        model.reset_classifier(0,'avg')


        if not hasattr(model, 'embed_dim'):
            x = model(torch.rand(1,in_chans,224,224))
            embed_dim = x.shape[1]
        else:
            embed_dim = model.embed_dim

    return model, embed_dim

def load_weights(
        model, 
        checkpoint_path ,
        nchannels=1,
        patch_size=14, 
        feat_dim=1024, 
        pos_embed_size=257, 
        ):

    # kernel_size = model.patch_embed.proj.kernel_size
    # stride = model.patch_embed.proj.stride
    # embed_dim = model.patch_embed.proj.out_channels # corresponds to embed_dim
    # print(model.pos_embed)

    # model.patch_embed.proj = nn.Conv2d(nchannels, feat_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    # model.pos_embed = nn.Parameter(torch.tensor(model.pos_embed[:, :pos_embed_size]))

    checkpoint = torch.load(checkpoint_path)
    if 'teacher' in checkpoint:
        d = checkpoint['teacher']
        d2 = OrderedDict([(k[9:], v) for k, v in d.items() if ('backbone' in k)])
        model.load_state_dict(d2, strict=False)
    if 'model' in checkpoint:
        d = checkpoint['model']
        d2 = OrderedDict([(k, v) for k, v in d.items() if ('decoder_blocks' not in k)])
        model.load_state_dict(d2, strict=False)

    return model

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for arch in ['dino_vitb16','dino_resnet50', 'efficientnet_b0','efficientnet_b3']:

        checkpoint_path = f'./logs/{arch}/checkpoint.pth'
        model, embed_dim = create_template_model(arch, in_chans=1)
        model = load_weights(model, checkpoint_path)
        model.to(device)

        print("Fit projection\n")
        fit_proj(
                model,
                size=224,
                nsamples=1_000,
                feat_dim=embed_dim,
                exclude_value=None,
                patch_w=16, 
                patch_h=16,
                method='umap',
                n_neighbors=20,
                cls_token=True,
                roi=None,
                batch_size=16,
                out_path_proj_model=f'./projs/{arch}.pkl'
                )


    """
    Inference of the fitted projection on the datatset following a grid.
    This results to a numpy array with pixels that are encompassing 
    several pixels in the original image: 
        - 16x16 pixels when working at patch_token level for a ViT with 16x16 patches
        - 224x224 when working at cls_token level
        
    """

    # Here an example performing inference only 
    # on the top left corner of the original image
    print(dataset.bounds)
    bb=dataset.bounds
    xlim = bb[0] + (bb[1]-bb[0])*0.25
    ylim = bb[2] + (bb[3]-bb[2])*0.25
    roi=BoundingBox(bb[0], xlim, bb[2], ylim, bb[4], bb[5])

    print("Projection inference\n")
    macro_img, bboxes = inf_proj(
            dataset,
            model,
            roi=roi,
            size=size,
            patch_w=16, 
            patch_h=16,
            cls_token=False,
            path_proj_model=f'{proj_dir}/proj.pkl',
            batch_size=500,
            )

    ## can be usefull to save progress
    # np.save(f'{proj_dir}/proj.npy', macro_img)

    numpy_to_geotif(
            original_image=os.path.join(naip_root,tile),
            numpy_image=macro_img,
            dtype='float32',
            pixel_scale=16,
            out_path=f'{proj_dir}/proj.tif'
            )

    # go back to the original spatial resolution if needed
    with rasterio.open(os.path.join(naip_root,tile)) as template_src:
        # Get the template resolution
        orig_resolution = template_src.transform[0]

    change_tif_resolution(f'{proj_dir}/proj.tif',f'{proj_dir}/proj_rescale.tif', orig_resolution)

