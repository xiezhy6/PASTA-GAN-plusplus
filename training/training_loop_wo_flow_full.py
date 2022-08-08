# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warnings
warnings.filterwarnings("ignore")

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from PIL import Image

import legacy
from metrics import metric_main

from util_functions import random_affine_matrix, get_affine_matrix, getRandomAffineParam, __patch_instance_norm_state_dict
import torch.nn.functional as F
import cv2
#----------------------------------------------------------------------------

def combine_parts(parts, col, row, gnum):
    col_part = parts[col]
    col_norm_img = col_part[:30,...]
    col_norm_img_upper = col_norm_img[:18,...]
    col_norm_img_lower = col_norm_img[18:,...]
    
    col_norm_pose = col_part[30:,...]
    col_norm_pose_upper = col_norm_pose[:18,...]
    col_norm_pose_lower = col_norm_pose[18:,...]

    row_part = parts[row]
    row_norm_img = row_part[:30,...]
    row_norm_img_upper = row_norm_img[:18,...]
    row_norm_img_lower = row_norm_img[18:,...]

    row_norm_pose = row_part[30:,...]
    row_norm_pose_upper = row_norm_pose[:18,...]
    row_norm_pose_lower = row_norm_pose[18:,...]

    gap = gnum // 3
    ##################### 换裤子
    if row < gap:
        combined_part = torch.cat([row_norm_img_upper,col_norm_img_lower,row_norm_pose_upper,col_norm_pose_lower], dim=0)
    ###################### 换全身
    elif row < 2 * gap:
        combined_part = col_part
    ###################### 换上衣
    else:
        combined_part = torch.cat([col_norm_img_upper,row_norm_img_lower,col_norm_pose_upper,row_norm_pose_lower], dim=0)

    return combined_part


def denorm_clothes(norm_patches, M_invs, norm_clothes_mask, col, row, gnum):
    denorm_upper_img = np.zeros((256,256,3),dtype=np.uint8)
    denorm_lower_img = np.zeros((256,256,3),dtype=np.uint8)
    # denorm_upper_clothes_mask = np.zeros((256,256,3),dtype=np.uint8)
    # denorm_lower_clothes_mask = np.zeros((256,256,3),dtype=np.uint8)

    kernel = np.ones((5,5),np.uint8)

    gap = gnum // 3
    for ii in range(M_invs.shape[1]):
        ################ 上半身
        if ii < 6:
            ####### 0-(gap-1)换裤子，上衣的索引是row
            if row < gap:
                norm_patch = norm_patches[row,ii*3:(ii+1)*3,...].transpose(1,2,0)
                norm_clothes_mask_patch = norm_clothes_mask[row,ii*3:(ii+1)*3,...].transpose(1,2,0)
                # M_inv = M_invs[row,ii]
            ###### gap-(2*gap-1)换全身，gap-(gum-1)换上衣，上衣的索引是col
            else:
                norm_patch = norm_patches[col,ii*3:(ii+1)*3,...].transpose(1,2,0)
                norm_clothes_mask_patch = norm_clothes_mask[col,ii*3:(ii+1)*3,...].transpose(1,2,0)
                # M_inv = M_invs[col,ii]
        ################## 下半身
        else:
            ###### 0-(gap-1)换裤子，gap-(2*gap-1)换全身，裤子索引是col 
            if row < 2 * gap:
                norm_patch = norm_patches[col,ii*3:(ii+1)*3,...].transpose(1,2,0)
                norm_clothes_mask_patch = norm_clothes_mask[col,ii*3:(ii+1)*3,...].transpose(1,2,0)
                # M_inv = M_invs[col,ii]
            ########## gap-(gum-1)换上衣，裤子的索引是row
            else:
                norm_patch = norm_patches[row,ii*3:(ii+1)*3,...].transpose(1,2,0)
                norm_clothes_mask_patch = norm_clothes_mask[row,ii*3:(ii+1)*3,...].transpose(1,2,0)
                # M_inv = M_invs[row,ii]
        M_inv = M_invs[row,ii]
        if M_inv.sum() == 0:
            continue
        denorm_patch = cv2.warpPerspective(norm_patch,M_inv,(256,256),borderMode=cv2.BORDER_CONSTANT)
        denorm_clothes_mask_patch = cv2.warpPerspective(norm_clothes_mask_patch,M_inv,(256,256),borderMode=cv2.BORDER_CONSTANT)
        denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)[...,0:1]
        denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)
        
        if ii < 6:
            denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)
        else:
            denorm_lower_img = denorm_patch * denorm_clothes_mask_patch + denorm_lower_img * (1-denorm_clothes_mask_patch)

    denorm_upper_img = denorm_upper_img.transpose(2,0,1)[np.newaxis,...]
    denorm_lower_img = denorm_lower_img.transpose(2,0,1)[np.newaxis,...]
    denorm_upper_clothes_mask = (np.sum(denorm_upper_img,axis=1,keepdims=True)>0).astype(np.uint8)
    denorm_lower_clothes_mask = (np.sum(denorm_lower_img,axis=1,keepdims=True)>0).astype(np.uint8)

    return denorm_upper_img, denorm_lower_img, denorm_upper_clothes_mask, denorm_lower_clothes_mask

def setup_snapshot_image_grid(training_set, device, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gnum = 23

    grid_indices = training_set.vis_index

    # Load data.
    images, poses, sems, norm_img, _, _, M_invs, _, _, _, _, norm_clothes_mask, retain_masks = zip(*[training_set[i] for i in grid_indices])
    
    print("load from dataset.")
    # process person image
    images = np.array(images)
    images = (torch.from_numpy(images).to(device).to(torch.float32) / 127.5 - 1)
    grid_ims = [images[i % gnum] for i in range(gnum * gnum)]
    grid_ims = torch.stack(grid_ims)

    norm_patches = np.array(norm_img)
    M_invs = np.array(M_invs)
    # denorm_random_mask = np.array(denorm_random_mask)
    norm_clothes_mask = np.array(norm_clothes_mask)

    # grid_denorm_ims, grid_denorm_masks, grid_denorm_clothes_masks = zip(*[denorm_clothes(norm_patches[i % gnum], M_invs[i//gnum], norm_clothes_mask[i % gnum], denorm_random_mask[i // gnum]) for i in range(gnum*gnum)])
    grid_denorm_upper_ims, grid_denorm_lower_ims, grid_denorm_upper_masks, grid_denorm_lower_masks = zip(*[denorm_clothes(norm_patches, M_invs, norm_clothes_mask, i % gnum, i // gnum, gnum) for i in range(gnum*gnum)])

    grid_denorm_upper_ims = np.concatenate(grid_denorm_upper_ims,axis=0)
    grid_denorm_upper_ims = torch.from_numpy(grid_denorm_upper_ims).to(device).to(torch.float32) / 127.5 - 1
    
    grid_denorm_lower_ims = np.concatenate(grid_denorm_lower_ims,axis=0)
    grid_denorm_lower_ims = torch.from_numpy(grid_denorm_lower_ims).to(device).to(torch.float32) / 127.5 - 1

    grid_denorm_upper_masks = np.concatenate(grid_denorm_upper_masks,axis=0)
    grid_denorm_upper_masks = torch.from_numpy(grid_denorm_upper_masks).to(device).to(torch.float32)

    grid_denorm_lower_masks = np.concatenate(grid_denorm_lower_masks,axis=0)
    grid_denorm_lower_masks = torch.from_numpy(grid_denorm_lower_masks).to(device).to(torch.float32)


    parts = np.array(norm_img)
    parts = torch.from_numpy(parts).to(device).to(torch.float32) / 127.5 - 1
    grid_norm_parts = [combine_parts(parts, i % gnum, i // gnum, gnum) for i in range(gnum * gnum)]
    grid_norm_parts = torch.stack(grid_norm_parts)

    # grid_norm_parts = [parts[i % gnum] for i in range(gnum * gnum)]
    # grid_norm_parts = torch.stack(grid_norm_parts)

    # process keypoints
    poses = np.array(poses)
    poses = (torch.from_numpy(poses).to(device).to(torch.float32) / 127.5 - 1)
    
    # process semantic parsing
    retain_masks = np.array(retain_masks)
    retain_masks = torch.from_numpy(retain_masks).to(device)

    sems = np.array(sems)
    sems = torch.from_numpy(sems).to(device)
    head_mask = (sems == 2).to(torch.uint8) + (sems == 13).to(torch.uint8) + \
                (sems == 1).to(torch.uint8) + (sems == 4).to(torch.uint8)
                # (sems == 9).to(torch.uint8) + (sems == 12).to(torch.uint8) + \
                # (sems == 16).to(torch.uint8) + (sems == 17).to(torch.uint8)
    head_mask = head_mask.unsqueeze(1)
    head_mask = head_mask + retain_masks
    head = head_mask * images - (1-head_mask)
    # head = head_mask.unsqueeze(1) * images - (1 - head_mask.unsqueeze(1))
    poses = torch.cat((poses, head), dim=1)
    grid_poses = [poses[i // gnum] for i in range(gnum * gnum)]
    grid_poses = torch.stack(grid_poses)

    # process feature wanted to retain
    retains = head.clone()
    grid_retains = [retains[i // gnum] for i in range(gnum * gnum)]
    grid_retains = torch.stack(grid_retains)

    return gnum, grid_ims, grid_poses, grid_norm_parts, grid_retains, grid_denorm_upper_ims, grid_denorm_lower_ims, \
             grid_denorm_upper_masks, grid_denorm_lower_masks

#----------------------------------------------------------------------------

def save_image_grid(im_side, im_top, img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    im_side = np.asarray(im_side, dtype=np.float32)
    im_side = (im_side - lo) * (255 / (hi - lo))
    im_side = np.rint(im_side).clip(0, 255).astype(np.uint8)

    im_top = np.asarray(im_top, dtype=np.float32)
    im_top = (im_top - lo) * (255 / (hi - lo))
    im_top = np.rint(im_top).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = np.reshape(img, (gh, gw, C, H, W))
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    img = np.concatenate((im_side, img), axis=1)
    img = np.concatenate((im_top, img), axis=0)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    # common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    common_kwargs = dict(c_dim=512, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, 30 * 2, 64, 64], device=device)
        c_feature = torch.empty([batch_gpu, 512], device=device)
        const_feature = torch.empty([batch_gpu, 3, 256, 256], device=device)
        pose = torch.empty([batch_gpu, 3+3, 256, 256], device=device)
        denorm_u = torch.empty([batch_gpu, 3, 256,256],device=device)
        denorm_l = torch.empty([batch_gpu, 3, 256,256],device=device)
        denorm_m_u = torch.empty([batch_gpu, 1, 256,256],device=device)
        denorm_m_l = torch.empty([batch_gpu, 1, 256,256],device=device)
        img = misc.print_module_summary(G, [z, c, const_feature, pose, denorm_u, denorm_l, denorm_m_u, denorm_m_l])
        misc.print_module_summary(D, [img[0], c_feature])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), 
                         ('G_const_encoding', G.const_encoding), ('G_style_encoding', G.style_encoding), 
                         ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    phases = []

    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    gnum = None
    grid_z = None
    grid_c = None
    grid_retain = None
    if rank == 0:
        print('Exporting sample images...')
        gnum, grid_im, grid_pose, grid_part, grid_retain, \
            grid_denorm_upper_im, grid_denorm_lower_im, grid_denorm_upper_mask, \
            grid_denorm_lower_mask = setup_snapshot_image_grid(training_set=training_set, 
                                                               device=device,
                                                               random_seed=int(time.time()) % 256)
        _N, C, H, W = grid_im.shape
        source_im = grid_im[:gnum].cpu()
        image_side = source_im.unsqueeze(1).numpy().transpose(0, 3, 1, 4, 2).reshape(gnum * H, 1 * W, C)
        image_top = torch.cat((torch.zeros(grid_im[0].shape).unsqueeze(0), source_im), dim=0)
        image_top = image_top.unsqueeze(0).numpy().transpose(0, 3, 1, 4, 2).reshape(1 * H, (gnum+1) * W, C)
        
        grid_poses = grid_pose.split(batch_gpu)
        grid_parts = grid_part.split(batch_gpu)
        grid_retains = grid_retain.split(batch_gpu)
        grid_denorm_upper_ims = grid_denorm_upper_im.split(batch_gpu)
        grid_denorm_lower_ims = grid_denorm_lower_im.split(batch_gpu)
        grid_denorm_upper_masks = grid_denorm_upper_mask.split(batch_gpu)
        grid_denorm_lower_masks = grid_denorm_lower_mask.split(batch_gpu)
        grid_z = torch.randn([grid_retain.shape[0], G.z_dim], device=device).split(batch_gpu)

        # grid_denorm_clothes_masks = grid_denorm_clothes_mask.split(batch_gpu)

        upper_images = torch.cat(grid_denorm_upper_ims).cpu().numpy()
        lower_images = torch.cat(grid_denorm_lower_ims).cpu().numpy()
        save_image_grid(image_side, image_top, upper_images, os.path.join(run_dir, 'init_denorm_upper.png'), drange=[-1,1], grid_size=(gnum, gnum))
        save_image_grid(image_side, image_top, lower_images, os.path.join(run_dir, 'init_denorm_lower.png'), drange=[-1,1], grid_size=(gnum, gnum))

        retain_images = torch.cat(grid_retains).cpu().numpy()
        save_image_grid(image_side, image_top, retain_images, os.path.join(run_dir, 'init_retain.png'), drange=[-1,1], grid_size=(gnum, gnum))

        # finetune_fake_images = torch.cat([G_ema(z=z, c=c, retain=retain, pose=pose, denorm_input=denorm_im, denorm_mask=denorm_ma, denorm_clothes_mask=denorm_cm, noise_mode='const')[1].cpu() for z, c, retain, denorm_im, denorm_ma, pose, denorm_cm \
                    # in zip(grid_z, grid_parts, grid_retains, grid_denorm_ims, grid_denorm_masks, grid_poses, grid_denorm_clothes_masks)]).numpy()
        # save_image_grid(image_side, image_top, finetune_fake_images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=(gnum, gnum))

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'w')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            # 读取模型输入数据和GT
            phase_real, phase_pose, phase_sem, phase_norm_img, phase_denorm_upper_img, phase_denorm_lower_img, _, \
                phase_upper_clothes_mask, phase_lower_clothes_mask, phase_denorm_upper_mask, phase_denorm_lower_mask, _, phase_retain_mask = next(training_set_iterator)
            
            phase_real_tensor = phase_real.to(device).to(torch.float32) / 127.5 - 1

            phase_sem = phase_sem.to(device)
            phase_retain_mask = phase_retain_mask.to(device)
            phase_parts_tensor = phase_norm_img.to(device).to(torch.float32) / 127.5 - 1

            phase_denorm_upper_img_tensor = phase_denorm_upper_img.to(device).to(torch.float32) / 127.5 - 1
            phase_denorm_lower_img_tensor = phase_denorm_lower_img.to(device).to(torch.float32) / 127.5 - 1
            # phase_denorm_random_mask_tensor = phase_denorm_random_mask.to(device).to(torch.float32)
            # phase_denorm_img_tensor = phase_denorm_img_tensor * (1-phase_denorm_random_mask_tensor) - phase_denorm_random_mask_tensor

            phase_denorm_upper_mask_tensor = phase_denorm_upper_mask.to(device).to(torch.float32)
            phase_denorm_lower_mask_tensor = phase_denorm_lower_mask.to(device).to(torch.float32)

            phase_pose_tensor = phase_pose.to(device).to(torch.float32) / 127.5 - 1
            phase_head_mask = (phase_sem == 2).to(torch.uint8) + (phase_sem == 13).to(torch.uint8) + \
                              (phase_sem == 1).to(torch.uint8) + (phase_sem == 4).to(torch.uint8)
            phase_head_mask = phase_head_mask.unsqueeze(1) + phase_retain_mask
            phase_head_tensor = phase_head_mask * phase_real_tensor - (1-phase_head_mask)
                            #    (phase_sem == 9).to(torch.uint8) + (phase_sem == 12).to(torch.uint8) + \
                            #    (phase_sem == 16).to(torch.uint8) + (phase_sem == 17).to(torch.uint8)
            # phase_head_tensor = phase_head_mask.unsqueeze(1) * phase_real_tensor - (1 - phase_head_mask.unsqueeze(1))
            phase_pose_tensor = torch.cat((phase_pose_tensor, phase_head_tensor), dim=1)

            phase_upper_clothes_mask_tensor = phase_upper_clothes_mask.to(device).to(torch.float32)
            phase_lower_clothes_mask_tensor = phase_lower_clothes_mask.to(device).to(torch.float32)

            # process head
            phase_retain_tensor = phase_head_tensor

            phase_real_tensor = phase_real_tensor.split(batch_gpu)
            phase_parts_tensor = phase_parts_tensor.split(batch_gpu)
            phase_pose_tensor = phase_pose_tensor.split(batch_gpu)                                           
            phase_retain_tensor = phase_retain_tensor.split(batch_gpu)

            phase_denorm_upper_img_tensor = phase_denorm_upper_img_tensor.split(batch_gpu)
            phase_denorm_lower_img_tensor = phase_denorm_lower_img_tensor.split(batch_gpu)

            phase_upper_clothes_mask_tensor = phase_upper_clothes_mask_tensor.split(batch_gpu)
            phase_lower_clothes_mask_tensor = phase_lower_clothes_mask_tensor.split(batch_gpu)

            phase_denorm_upper_mask_tensor = phase_denorm_upper_mask_tensor.split(batch_gpu)
            phase_denorm_lower_mask_tensor = phase_denorm_lower_mask_tensor.split(batch_gpu)

            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            del phase_real      # conserve memory
            del phase_pose       # conserve memory
            del phase_sem        # conserve memory
            del phase_head_mask   # conserve memory

        # Execute training phases.
        for phase, phase_gen_z in zip(phases, all_gen_z):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, gen_z, style_input, retain, pose, denorm_upper_input, denorm_lower_input, \
                            denorm_upper_mask, denorm_lower_mask, upper_c_mask, lower_c_mask) \
                    in enumerate(zip(phase_real_tensor, phase_gen_z, phase_parts_tensor, \
                                     phase_retain_tensor, phase_pose_tensor, phase_denorm_upper_img_tensor,\
                                     phase_denorm_lower_img_tensor, phase_denorm_upper_mask_tensor, \
                                     phase_denorm_lower_mask_tensor, phase_upper_clothes_mask_tensor, \
                                     phase_lower_clothes_mask_tensor)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                # 把style_input当做 real_c 和 gen_c。为了增加可变性, gen_z还是保留
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, gen_z=gen_z, style_input=style_input, 
                                          retain=retain, pose=pose, denorm_upper_input=denorm_upper_input, 
                                          denorm_lower_input=denorm_lower_input, denorm_upper_mask=denorm_upper_mask,
                                          denorm_lower_mask=denorm_lower_mask, upper_clothes_mask=upper_c_mask, 
                                          lower_clothes_mask=lower_c_mask, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        # 这里可以修改下输入，改为换衣sampling
        # 第一行是 source top and pant
        # 第一列是 reference person
        # 2, 3, 4 行是 换top
        # 4, 5, 6 行是 换pant
        # 7, 8 第一行的reconstruction

        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            finetune_fake_images = torch.cat([G_ema(z=z, c=c, retain=retain, pose=pose, denorm_upper_input=denorm_upper_im, denorm_lower_input=denorm_lower_im, \
                                    denorm_upper_mask=denorm_upper_ma, denorm_lower_mask=denorm_lower_ma, noise_mode='const')[1].cpu() for z, c, retain, denorm_upper_im, \
                                    denorm_lower_im, denorm_upper_ma, denorm_lower_ma, pose \
                    in zip(grid_z, grid_parts, grid_retains, grid_denorm_upper_ims, grid_denorm_lower_ims, grid_denorm_upper_masks, grid_denorm_lower_masks, grid_poses)]).numpy()
            save_image_grid(image_side, image_top, finetune_fake_images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_finetune.png'), drange=[-1,1], grid_size=(gnum, gnum))

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        #     if rank == 0:
        #         print('Evaluating metrics...')
        #     for metric in metrics:
        #         result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #         if rank == 0:
        #             metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #         stats_metrics.update(result_dict.results)
        # del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
