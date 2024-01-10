# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import itertools
from functools import partial
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import timm
from timm import create_model

from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples, BoundingBox
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.transforms import AugmentationSequential
import kornia.augmentation as K
from kornia.enhance.normalize import Normalize
import rasterio

import utils
import vision_transformer as vits
from vision_transformer import DINOHead


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='dino_vitb16', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/home/ptresson/congo/panchro_congo_all_renamed', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./logs/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # torchgeo specific
    parser.add_argument("--filename_glob", default="*.[tT][iI][fF]", type=str, help="Filename glob to select dataset files")
    parser.add_argument("--file_path", default="./data/tif/", type=str, help="directory containing the raster files")
    parser.add_argument("--sample_size", default=224, type=int, help="size of samples (px) in torchgeo")
    parser.add_argument("--num_samples", default=10_000, type=int, help="number of sample for torchgeo sampler")
    parser.add_argument("--mean_dataset", default=[3000], type=int, help="number of sample for torchgeo sampler")
    parser.add_argument("--sd_dataset", default=[1000], type=int, help="number of sample for torchgeo sampler")

    return parser


def collate_data_and_cast_torchgeo(samples_list):

    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    collated_global_crops = torch.stack([s["global_crops"][i]['image'][0] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops = torch.stack([s["local_crops"][i]['image'][0] for i in range(n_local_crops) for s in samples_list])

    return {
        "collated_global_crops": collated_global_crops,     #.to(dtype)
        "collated_local_crops": collated_local_crops,       #.to(dtype)
    }


class MergedDataLoader:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
        self.dataset = None  # You may need to define a custom dataset for this

    def __iter__(self):
        return itertools.chain(*[iter(dataloader) for dataloader in self.dataloaders])

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloaders)


def vit_first_layer_with_nchan(
        model,
        in_chans=1,
        embed_dim=768, 
        patch_size=16,
        ):

    # cf. https://github.com/facebookresearch/dino/issues/214
    # create empty proj layer
    new_conv = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    weight = model.patch_embed.proj.weight.clone()
    bias = model.patch_embed.proj.bias.clone()
    with torch.no_grad():
        for i in range(0,in_chans):
            j = i%3 # cycle every 3 bands
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
        new_conv.bias[:] = bias[:]
    model.patch_embed.proj = new_conv

    return model


def resnet_first_layer_with_nchan(
        model,
        in_chans=1,
        ):

    # cf. https://github.com/facebookresearch/dino/issues/214
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    new_conv = torch.nn.Conv2d(in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    weight = model.conv1.weight.clone()
    with torch.no_grad():
        for i in range(0,in_chans):
            j = i%3 # cycle every 3 bands
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
    model.conv1 = new_conv

    return model


def get_model(arch, in_chans, drop_path_rate, pretrained=True, patch_size=16):

    if arch in torch.hub.list("facebookresearch/dino:main"):
        if 'vit' in arch :
            student = torch.hub.load('facebookresearch/dino:main', arch,
                                    drop_path_rate=drop_path_rate,
                                    in_chans=in_chans,
                                    strict=False,
                                    pretrained=False,
                                    )
            teacher = torch.hub.load('facebookresearch/dino:main', 
                    arch, 
                    in_chans=in_chans,
                    pretrained=False,
                    )

        if 'resnet' in arch :
            print("================== Resnet50")
            student = torch.hub.load('facebookresearch/dino:main', arch,
                                    )
            student = resnet_first_layer_with_nchan(student, in_chans)

            teacher = torch.hub.load('facebookresearch/dino:main', 
                    arch, 
                    )
            teacher = resnet_first_layer_with_nchan(teacher, in_chans)
        # teacher = torch.hub.load('facebookresearch/dino:main', 
        #         arch, 
        #         in_chans=in_chans,
        #         pretrained=False,
        #         )

        # get embed_dim before fully loading model to avoid hardcoding value
        if not hasattr(student, 'embed_dim'):
            x = student(torch.rand(1,in_chans,224,224))
            student.fc, student.head = nn.Identity(), nn.Identity()
            embed_dim = x.shape[1]
        else:
            embed_dim = student.embed_dim


        if pretrained and in_chans != 3:
            if 'vit' in arch :
                pretrained = torch.hub.load('facebookresearch/dino:main', arch,
                                        pretrained=True, 
                                        drop_path_rate=drop_path_rate,
                                        )
                pretrained=vit_first_layer_with_nchan(
                        pretrained,
                        in_chans=in_chans,
                        embed_dim=embed_dim,
                        patch_size=patch_size
                        )
            if 'resnet' in arch :
                pretrained = torch.hub.load('facebookresearch/dino:main', arch,
                                        pretrained=True, 
                                        )
                pretrained=resnet_first_layer_with_nchan(
                        pretrained,
                        in_chans=in_chans,
                        )

            student.load_state_dict(pretrained.state_dict())
            teacher.load_state_dict(pretrained.state_dict())



    elif arch in timm.list_models(pretrained=True):
        student = create_model(
                arch,
                pretrained=pretrained,
                drop_path_rate=drop_path_rate,
                in_chans=in_chans,
                )
        teacher = create_model(
                arch,
                pretrained=pretrained,
                in_chans=in_chans,
                )

        if not hasattr(student, 'embed_dim'):
            x = student(torch.rand(1,in_chans,224,224))
            student.fc, student.head = nn.Identity(), nn.Identity()
            embed_dim = x.shape[1]
        else:
            embed_dim = student.embed_dim

    return student, teacher, embed_dim



def train_dino(args, data_loader):

    student, teacher, embed_dim = get_model(args.arch, len(args.mean_dataset), args.drop_path_rate)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        # data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        global_crops = batch['collated_global_crops']
        local_crops = batch['collated_local_crops']
        global_crops = global_crops.cuda(non_blocking=True)
        local_crops = local_crops.cuda(non_blocking=True)
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        # images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(global_crops)  # only the 2 global views pass through the teacher
            student_output = student(local_crops)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student_out = student_output / self.student_temp
        # student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        # total_loss = 0
        # n_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        #         if v == iq:
        #             # we skip cases where student and teacher operate on the same view
        #             continue
        #         loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #         total_loss += loss.mean()
        #         n_loss_terms += 1
        # total_loss /= n_loss_terms
        # self.update_center(teacher_output)
        # return total_loss

        # taken from dinov2
        total_loss = 0
        for s in student_output:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class GaussianBlur(transforms.RandomApply):
    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = K.RandomGaussianBlur(kernel_size=9, sigma=(radius_min, radius_max), p=p)     
        super().__init__(transforms=[transform], p=keep_p)


class DataAugmentationDINOTorchgeo(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        dataset_mean = [0],
        dataset_std = [1]
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        
        # random resized crop and flip
        self.geometric_augmentation_global = AugmentationSequential(
                #K.RandomResizedCrop(size=global_crops_size, scale=global_crops_scale),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["image"]
        )

        self.geometric_augmentation_local = AugmentationSequential(
                #K.RandomResizedCrop(size=local_crops_size, scale=local_crops_scale),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["image"]
        )

        # normalization
        self.normalize = AugmentationSequential(
                Normalize(dataset_mean, dataset_std),
                data_keys=["image"],
        )
      
        self.global_transfo1 = AugmentationSequential(GaussianBlur(p=1.0), data_keys=["image"])
        self.global_transfo2 = AugmentationSequential(GaussianBlur(p=0.1), 
                                                       transforms.RandomSolarize(threshold=0, p=0.2), 
                                                        data_keys=["image"])
        self.local_transfo = AugmentationSequential(GaussianBlur(p=0.5), data_keys=["image"])
        

    def __call__(self, image):
        output = {}

        image = self.normalize(image)
        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)
        output["global_crops"] = [global_crop_1, global_crop_2]
        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]
        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

def prepare_congo_data(args):

    transform = DataAugmentationDINOTorchgeo(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        dataset_mean = args.mean_dataset,
        dataset_std = args.sd_dataset,
    )

    class Raster(RasterDataset):
        filename_glob = args.filename_glob
        is_image = True

    dataset1 = Raster(os.path.join(args.data_path,'A'))
    dataset2 = Raster(os.path.join(args.data_path,'B'))

    dataset1.transforms = transform
    dataset2.transforms = transform

    bb1 = dataset1.index.bounds
    bb2 = dataset2.index.bounds
    roi1 = BoundingBox(bb1[0], bb1[1], bb1[2], bb1[3], bb1[4], bb1[5])
    roi2 = BoundingBox(bb2[0], bb2[1], bb2[2], bb2[3], bb2[4], bb2[5])

    sampler1 = RandomGeoSampler(
            dataset1, 
            size=(args.sample_size,args.sample_size), 
            length=args.num_samples, 
            roi=roi1
            )
    sampler2 = RandomGeoSampler(
            dataset2, 
            size=(args.sample_size,args.sample_size), 
            length=args.num_samples, 
            roi=roi2
            )

    collate_fn = partial(
        collate_data_and_cast_torchgeo,
    )

    data_loader1 = DataLoader(
            dataset1, 
            sampler=sampler1, 
            collate_fn=collate_fn, 
            shuffle=False, 
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            )
    data_loader2 = DataLoader(
            dataset2, 
            sampler=sampler2, 
            collate_fn=collate_fn, 
            shuffle=False, 
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            )

    data_loader=MergedDataLoader(data_loader1, data_loader2)

    return data_loader


def get_mean_sd_by_band_gdal(tif, ignore_zeros=True):
    import subprocess
    import re

    # Run the gdalinfo command and capture the output
    command = f'gdalinfo -mm -stats {tif}'
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)

    # Extract the STATISTICS_MEAN value using regex
    mean_match = re.search(r'STATISTICS_MEAN=([\d.]+)', result.stdout)
    mean_value = float(mean_match.group(1))
    sd_match = re.search(r'STATISTICS_STDDEV=([\d.]+)', result.stdout)
    sd_value = float(sd_match.group(1))

    return mean_value, sd_value


def get_mean_sd_by_band(tif, ignore_zeros=True):
    '''
    reads metadata or computes mean and sd of each band of a geotif
    '''

    src = rasterio.open(tif)
    means = []
    sds = []
    npx = src.width * src.height

    for band in range(1, src.count+1):

        if src.tags(band) != {}: # if metadata are available
            try:
                mean = src.tags(band)['STATISTICS_MEAN']
                sd = src.tags(band)['STATISTICS_STDDEV']
            except (KeyError, IndexError):
                mean, sd = get_mean_sd_by_band_gdal(tif)

        else: # if not, just compute it
            if ignore_zeros:
                arr = src.read(band)
                mean = np.ma.masked_equal(arr, 0).mean()
                sd = np.ma.masked_equal(arr, 0).std()
                del arr # cleanup memory in doubt

            else:    
                arr = src.read(band)
                mean = np.mean(arr)
                sd = np.std(arr)
                del arr # cleanup memory in doubt

        means.append(float(mean))
        sds.append(float(sd))
        npx

    src.close()
    return  means, sds, npx

def get_dataset_mean_sd(args):

    pathA = os.path.join(args.data_path,'A/')
    pathB = os.path.join(args.data_path,'B/')
    filesA = [os.path.join(pathA, f) for f in os.listdir(pathA) if f.lower().endswith(('.tif','tiff'))]
    filesB = [os.path.join(pathB, f) for f in os.listdir(pathB) if f.lower().endswith(('.tif','tiff'))]
    files = filesA + filesB
    files = [f for f in files if not f.endswith(('8B.tif','4A.tif'))] # have saturated pixels that change the distribution

    tot_means = []
    tot_sds = []
    tot_npx = []

    for file in files:
        print(file)
        means, sds, npx = get_mean_sd_by_band(file)
        print(means, sds, npx)
        tot_means.append(means[0])
        tot_sds.append(sds[0])
        tot_npx.append(npx)

    tot_means = np.array(tot_means)
    tot_sds = np.array(tot_sds)
    tot_npx = np.array(tot_npx)

    # Compute weighted mean
    weighted_mean = np.sum(tot_means * tot_npx) / np.sum(tot_npx)

    # Compute weighted standard deviation
    weighted_sd = np.sqrt(np.sum(tot_npx * (tot_sds ** 2 + (tot_means - weighted_mean) ** 2)) / np.sum(tot_npx))

    print("Weighted Mean:", weighted_mean)
    print("Weighted Standard Deviation:", weighted_sd)



    sys.exit(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    ## Operation is done one time and then hardcoded here
    # get_dataset_mean_sd(args)

    data_loader = prepare_congo_data(args)

    for arch in ['dino_vitb16','dino_resnet50', 'efficientnet_b0','efficientnet_b3']:
        args.output_dir = f'./logs/{arch}' 
        args.arch = arch
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_dino(args, data_loader)
