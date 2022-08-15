# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn as nn

from training import dataset as custom_dataset

import legacy
import cv2
import tqdm

import scipy.io as sio
import tqdm

CMAP = sio.loadmat('human_colormap.mat')['colormap']
CMAP = (CMAP * 256).astype(np.uint8)

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--dataroot',type=str)
@click.option('--batchsize',type=int)
@click.option('--testpart',type=str)
@click.option('--testtxt',type=str)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    dataroot: str,
    batchsize: str,
    testpart: str,
    testtxt: str
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    if testpart == 'full':
        dataset = custom_dataset.UvitonDatasetFull_512_test_full(path=dataroot,test_txt=testtxt,use_labels=True, max_size=None, xflip=False)
    elif testpart == 'upper':
        dataset = custom_dataset.UvitonDatasetFull_512_test_upper(path=dataroot,test_txt=testtxt,use_labels=True, max_size=None, xflip=False)
    elif testpart == 'lower':
        dataset = custom_dataset.UvitonDatasetFull_512_test_lower(path=dataroot,test_txt=testtxt,use_labels=True, max_size=None, xflip=False)
    else:
        raise ValueError('Invalid value for test part!')
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=False,pin_memory=True, num_workers=0)

    device = torch.device('cuda')

    for data in tqdm.tqdm(dataloader):
        image, clothes, pose, _, norm_img, norm_img_lower, denorm_upper_clothes, denorm_lower_clothes, \
            denorm_upper_mask, denorm_lower_mask, retain_mask, skin_average, lower_label_map, lower_clothes_upper_bound, \
            person_name, clothes_name = data

        image_tensor = image.to(device).to(torch.float32) / 127.5 - 1
        clothes_tensor = clothes.to(device).to(torch.float32) / 127.5 - 1
        pose_tensor = pose.to(device).to(torch.float32) / 127.5 - 1
        norm_img_tensor = norm_img.to(device).to(torch.float32) / 127.5 - 1
        norm_img_lower_tensor = norm_img_lower.to(device).to(torch.float32) / 127.5 - 1
        
        skin_tensor = skin_average.to(device).to(torch.float32) / 127.5 - 1
        lower_label_map_tensor = lower_label_map.to(device).to(torch.float32) / 127.5 - 1
        lower_clothes_upper_bound_tensor = lower_clothes_upper_bound.to(device).to(torch.float32) / 127.5 - 1

        parts_tensor = torch.cat([norm_img_tensor, norm_img_lower_tensor],dim=1)

        denorm_upper_clothes_tensor = denorm_upper_clothes.to(device).to(torch.float32) / 127.5 - 1
        denorm_upper_mask_tensor = denorm_upper_mask.to(device).to(torch.float32)

        denorm_lower_clothes_tensor = denorm_lower_clothes.to(device).to(torch.float32) / 127.5 - 1
        denorm_lower_mask_tensor = denorm_lower_mask.to(device).to(torch.float32)

        retain_mask_tensor = retain_mask.to(device)
        retain_tensor = image_tensor * retain_mask_tensor - (1-retain_mask_tensor)
        pose_tensor = torch.cat([pose_tensor,lower_label_map_tensor,lower_clothes_upper_bound_tensor],dim=1)
        retain_tensor = torch.cat([retain_tensor,skin_tensor],dim=1)
        gen_z = torch.randn([batchsize,0],device=device)

        with torch.no_grad():
            gen_c, cat_feat_list = G.style_encoding(parts_tensor, retain_tensor)
            pose_feat = G.const_encoding(pose_tensor)
            ws = G.mapping(gen_z,gen_c)
            cat_feats = {}
            for cat_feat in cat_feat_list:
                h = cat_feat.shape[2]
                cat_feats[str(h)] = cat_feat
            gt_parsing = None
            _, gen_imgs, _ = G.synthesis(ws, pose_feat, cat_feats, denorm_upper_clothes_tensor, denorm_lower_clothes_tensor, \
                                        denorm_upper_mask_tensor, denorm_lower_mask_tensor, gt_parsing)

        for ii in range(gen_imgs.size(0)):
            gen_img = gen_imgs[ii].detach().cpu().numpy()
            gen_img = (gen_img.transpose(1,2,0)+1.0) * 127.5
            gen_img = np.clip(gen_img,0,255)
            gen_img = gen_img.astype(np.uint8)[...,[2,1,0]]

            image_np = image_tensor[ii].detach().cpu().numpy()
            image_np = (image_np.transpose(1,2,0)+1.0) * 127.5
            image_np = image_np.astype(np.uint8)[...,[2,1,0]]

            clothes_np = clothes_tensor[ii].detach().cpu().numpy()
            clothes_np = (clothes_np.transpose(1,2,0)+1.0) * 127.5
            clothes_np = clothes_np.astype(np.uint8)[...,[2,1,0]]

            result = np.concatenate([clothes_np[:,96:416,:], image_np[:,96:416,:], \
                                     gen_img[:,96:416,:]], axis=1)

            person_n = person_name[ii].split('/')[-1]
            clothes_n = clothes_name[ii].split('/')[-1]
                                
            save_name = person_n[:-4]+'___'+clothes_n[:-4]+'.png'
            save_path = os.path.join(outdir, save_name)
            cv2.imwrite(save_path,result)

    print('finish')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
