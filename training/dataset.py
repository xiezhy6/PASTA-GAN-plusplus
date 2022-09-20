# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#coding:utf-8

from curses import noecho
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2
from skimage.draw import circle, line_aa

from training.utils import get_hand_mask, get_palm_mask
import math
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
try:
    import pyspng
except ImportError:
    pyspng = None

import random
import pickle
import glob

MISSING_VALUE = -1
# LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
#            [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
#            [0,15], [15,17], [2,16], [5,17]]

# COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
#           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
#           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

kptcolors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0]]

limbseq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    # def _get_raw_labels(self):
    #     if self._raw_labels is None:
    #         self._raw_labels = self._load_raw_labels() if self._use_labels else None
    #         if self._raw_labels is None:
    #             self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
    #         assert isinstance(self._raw_labels, np.ndarray)
    #         assert self._raw_labels.shape[0] == self._raw_shape[0]
    #         assert self._raw_labels.dtype in [np.float32, np.int64]
    #         if self._raw_labels.dtype == np.int64:
    #             assert self._raw_labels.ndim == 1
    #             assert np.all(self._raw_labels >= 0)
    #     return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_person_parts_image(self, raw_idx, image, keypoints): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_pose_heatmap(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    # def _load_raw_labels(self): # to be overridden by subclass
    #     raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        person_img = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(person_img, np.ndarray)
        assert list(person_img.shape) == self.image_shape
        assert person_img.dtype == np.uint8
        pose_heatmap, keypoints = self._load_raw_pose_heatmap(self._raw_idx[idx])
        head_img, top_img, pant_img, palm_img = self._load_person_parts_image(self._raw_idx[idx], person_img, keypoints)

        if self._xflip[idx]:
            assert person_img.ndim == 3 # CHW
            person_img = person_img[:, :, ::-1]
            pose_heatmap = pose_heatmap[:, :, ::-1]
            head_img = head_img[:, :, ::-1]
            top_img = top_img[:, :, ::-1]
            pant_img = pant_img[:, :, ::-1]
            palm_img = palm_img[:, :, ::-1]

        return person_img.copy(), pose_heatmap.copy(), head_img.copy(), top_img.copy(), pant_img.copy(), palm_img.copy()

    def get_label(self, idx):
        person_img = self._load_raw_image(self._raw_idx[idx])
        pose_heatmap, keypoints = self._load_raw_pose_heatmap(self._raw_idx[idx])
        head_img, top_img, pant_img, palm_img = self._load_person_parts_image(self._raw_idx[idx], person_img, keypoints)

        return pose_heatmap.copy(), head_img.copy(), top_img.copy(), pant_img.copy(), palm_img.copy()

    # def get_label(self, idx):
    #     label = self._get_raw_labels()[self._raw_idx[idx]]
    #     if label.dtype == np.int64:
    #         onehot = np.zeros(self.label_shape, dtype=np.float32)
    #         onehot[label] = 1
    #         label = onehot
    #     return label.copy()

    # def get_details(self, idx):
    #     d = dnnlib.EasyDict()
    #     d.raw_idx = int(self._raw_idx[idx])
    #     d.xflip = (int(self._xflip[idx]) != 0)
    #     d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
    #     return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def vis_index(self):
        return self._vis_index

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW

        return image
    
    def _load_person_parts_image(self, raw_idx, person_img, keypoints):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            image_paths = json.load(f)['image_paths']

        parsing_path = image_paths[raw_idx].replace('.jpg', '_label.png')
        if parsing_path.find("/deepfashion/") > -1:
            parsing_path = parsing_path.replace("/img_320_512_image/", "/img_320_512_parsing/")
        elif parsing_path.find("/mpv/") > -1:
            parsing_path = parsing_path.replace("/MPV_320_512_image/", "/MPV_320_512_parsing/")
        elif parsing_path.find("/Zalando_512_320/") > -1:
            parsing_path = parsing_path.replace("/image/", "/parsing/")
        elif parsing_path.find("/Zalora_512_320/") > -1:
            parsing_path = parsing_path.replace("/image/", "/parsing/")

        parsing_label = cv2.imread(parsing_path)[...,0:1]
        head_mask = (parsing_label == 2).astype(np.float32) + (parsing_label == 13).astype(np.float32)
        top_mask = (parsing_label == 5).astype(np.float32) + (parsing_label == 6).astype(np.float32) + (parsing_label == 7).astype(np.float32) + (parsing_label == 11).astype(np.float32)
        pant_mask = (parsing_label == 8).astype(np.float32) + (parsing_label == 9).astype(np.float32) + (parsing_label == 12).astype(np.float32) + (parsing_label == 18).astype(np.float32) + (parsing_label == 19).astype(np.float32)

        # palm mask from keypoint
        left_hand_keypoints = keypoints[[5,6,7],:]
        right_hand_keypoints = keypoints[[2,3,4],:]
        height, width, _ = parsing_label.shape
        left_hand_up_mask, left_hand_botton_mask = get_hand_mask(left_hand_keypoints, height, width)
        right_hand_up_mask, right_hand_botton_mask = get_hand_mask(right_hand_keypoints, height, width)
        # palm mask refined by parsing
        left_hand_mask = (parsing_label == 14)
        right_hand_mask = (parsing_label == 15)
        left_palm_mask = get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = (left_palm_mask + right_palm_mask)

        h, w, _ = parsing_label.shape
        if h > w:
            w_x_left = (int) ((h - w) / 2)
            w_x_right = h - w - w_x_left
            head_mask = np.pad(head_mask, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
            top_mask = np.pad(top_mask, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
            pant_mask = np.pad(pant_mask, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
            palm_mask = np.pad(palm_mask, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
        elif h < w:
            w_y_top = (int) ((w - h) / 2)
            w_y_bottom = w - h - w_y_top
            head_mask = np.pad(head_mask, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
            top_mask = np.pad(top_mask, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
            pant_mask = np.pad(pant_mask, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
            palm_mask = np.pad(palm_mask, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
        
        head_mask = head_mask.transpose(2, 0, 1) # HWC => CHW
        top_mask = top_mask.transpose(2, 0, 1)   # HWC => CHW
        pant_mask = pant_mask.transpose(2, 0, 1) # HWC => CHW
        palm_mask = palm_mask.transpose(2, 0, 1) # HWC => CHW

        head_mask = head_mask > 0
        top_mask = top_mask > 0
        pant_mask = pant_mask > 0
        palm_mask = palm_mask > 0

        head_img = person_img * head_mask
        top_img = person_img * top_mask
        pant_img = person_img * pant_mask
        palm_img = person_img * palm_mask

        

        return head_img, top_img, pant_img, palm_img

    def _load_raw_pose_heatmap(self, raw_idx):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            image_paths = json.load(f)['image_paths']

        pose_path = image_paths[raw_idx].replace('.jpg', '_keypoints.json')
        if pose_path.find("/deepfashion/") > -1:
            pose_path = pose_path.replace("/img_320_512_image/", "/img_320_512_keypoints/")
        elif pose_path.find("/mpv/") > -1:
            pose_path = pose_path.replace("/MPV_320_512_image/", "/MPV_320_512_keypoints/")
        elif pose_path.find("/Zalando_512_320/") > -1:
            pose_path = pose_path.replace("/image/", "/keypoints/")
        elif pose_path.find("/Zalora_512_320/") > -1:
            pose_path = pose_path.replace("/image/", "/keypoints/")

        heatmap, keypoints = self.get_pose_heatmaps(pose_path)

        return heatmap, keypoints

    # def _load_raw_labels(self):
    #     fname = 'dataset.json'
    #     if fname not in self._all_fnames:
    #         return None
    #     with self._open_file(fname) as f:
    #         labels = json.load(f)['labels']
    #     if labels is None:
    #         return None
    #     labels = dict(labels)
    #     labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
    #     labels = np.array(labels)
    #     labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
    #     return labels

    def cords_to_map(self, cords, img_size=(512, 320), sigma=8):
        result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
        for i, point in enumerate(cords):
            if point[2] == -1:
                continue
            x_matrix, y_matrix = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            # result[..., i] = np.exp(-((x_matrix - int(point[1])) ** 2 + (y_matrix - int(point[0])) ** 2) / (2 * sigma ** 2))
            # result[..., i] = np.exp(-((x_matrix - point[0]) ** 2 + (y_matrix - point[1]) ** 2) / (2 * sigma ** 2))
            result[..., i] = np.where(((x_matrix - point[0]) ** 2 + (y_matrix - point[1]) ** 2) < (sigma ** 2), 1, 0) # ???�?�?�????��?��??1�?7

        # padding ???1�?7?512
        h, w, c = result.shape # (H, W, C)
        if h > w:
            w_x_left = (int) ((h - w) / 2)
            w_x_right = h - w - w_x_left
            result = np.pad(result, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
        elif h < w:
            w_y_top = (int) ((w - h) / 2)
            w_y_bottom = w - h - w_y_top
            result = np.pad(result, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
        result = result.transpose(2, 0, 1) # HWC => CHW

        return result

    def get_pose_heatmaps(self, pose_path):
        datas = None
        with open(pose_path, 'r') as f:
            datas = json.load(f)
        keypoints = np.array(datas['people'][0]['pose_keypoints_2d']).reshape((-1,3))
        for i in range(keypoints.shape[0]):
            if keypoints[i, 0] <= 0 or keypoints[i,1] <= 0:
                keypoints[i, 2] = -1
            if keypoints[i, 2] < 0.01:
                keypoints[i, 2] = -1
        pose_heatmap = self.cords_to_map(keypoints)

        return pose_heatmap, keypoints

#----------------------------------------------------------------------------
          

class UvitonDatasetFull_512(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir' 
            dataset_list = ['Zalando_512_320_v1', 'Zalando_512_320_v2', 
                            'Zalora_512_320_v1', 'Zalora_512_320_v2',
                            'Deepfashion_512_320', 'MPV_512_320', 
                            'ZMO_dresses_512_320', 'Zalando_512_320_v1_flip',
                            'Zalando_512_320_v2_flip', 'Zalora_512_320_v1_flip',
                            'Zalora_512_320_v2_flip', 'Deepfashion_512_320_flip',
                            'MPV_512_320_flip', 'ZMO_dresses_512_320_flip']

            self._image_fnames = []
            self._kpt_fnames = []
            self._parsing_fnames = []
            self._garment_parsing_fnames = []
            for dataset in dataset_list:
                txt_path = os.path.join(self._path, dataset, 'train_pairs_front_list_220508.txt')
                with open(txt_path, 'r') as f:
                    for person in f.readlines():
                        person = person.strip().split()[0]
                        self._image_fnames.append(os.path.join(dataset,'image',person))
                        self._kpt_fnames.append(os.path.join(dataset,'keypoints',person.replace('.jpg', '_keypoints.json')))
                        self._garment_parsing_fnames.append(os.path.join(dataset,'garment_parsing',person.replace('.jpg','.png')))
                        if dataset == 'Deepfashion_512_320' or dataset == 'MPV_512_320':
                            self._parsing_fnames.append(os.path.join(dataset,'parsing',person.replace('.jpg','_label.png')))
                        else:
                            self._parsing_fnames.append(os.path.join(dataset,'parsing',person.replace('.jpg','.png')))

            index_list = list(range(len(self._image_fnames)))
            random.shuffle(index_list)
            self._image_fnames = [self._image_fnames[index] for index in index_list]
            self._kpt_fnames = [self._kpt_fnames[index] for index in index_list]
            self._parsing_fnames = [self._parsing_fnames[index] for index in index_list]
            self._garment_parsing_fnames = [self._garment_parsing_fnames[index] for index in index_list]

            vis_dir = os.path.join(self._path,'train_img_front_vis_512_220414')
            image_list = sorted(os.listdir(vis_dir))
            vis_index = []
            for image_name in image_list:
                deepfashion_path = os.path.join(self._path, 'Deepfashion_512_320', 'image', 'train', image_name)
                zalando_path = os.path.join(self._path, 'Zalando_512_320_v1', 'image', image_name)
                zalora_path = os.path.join(self._path, 'Zalora_512_320_v2', 'image', image_name)
                if os.path.exists(zalando_path):
                    vis_index.append(self._image_fnames.index(os.path.join('Zalando_512_320_v1','image', image_name)))
                elif os.path.exists(deepfashion_path):
                    vis_index.append(self._image_fnames.index(os.path.join('Deepfashion_512_320','image', 'train', image_name)))
                elif os.path.exists(zalora_path):
                    vis_index.append(self._image_fnames.index(os.path.join('Zalora_512_320_v2', 'image', image_name)))

            self._vis_index = vis_index

            random_mask_acgpn_dir = os.path.join(self._path, 'train_random_mask_acgpn')
            self._random_mask_acgpn_fnames = [os.path.join(random_mask_acgpn_dir, mask_name) for mask_name in os.listdir(random_mask_acgpn_dir)]
            self._mask_acgpn_numbers = len(self._random_mask_acgpn_fnames)
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        im_shape = list((self._load_raw_image(0))[0].shape)
        raw_shape = [len(self._image_fnames)] + [im_shape[2], im_shape[0], im_shape[1]]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        # load images --> range [0, 255]
        fname = self._image_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        self.image = np.array(PIL.Image.open(f))
        im_shape = self.image.shape
        h, w = im_shape[0], im_shape[1]
        left_padding = (h-w) // 2
        right_padding = h-w-left_padding
        image = np.pad(self.image,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        # load keypoints --> range [0, 1]
        fname = self._kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        pose, keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        pose = np.pad(pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        keypoints[:,0] += left_padding

        # load garment parsing
        fname = self._garment_parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        garment_parsing = cv2.imread(f)[...,0:1]
        garment_parsing = np.pad(garment_parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))
        sleeve_mask = (garment_parsing==10).astype(np.uint8) + (garment_parsing==11).astype(np.uint8)

        # load upper_cloth and lower body
        fname = self._parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        parsing = cv2.imread(f)[...,0:1]
        parsing = np.pad(parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))

        shoes_mask = (parsing==18).astype(np.uint8) + (parsing==19).astype(np.uint8)
        head_mask = (parsing==1).astype(np.uint8) + (parsing==2).astype(np.uint8) + \
                    (parsing==4).astype(np.uint8) + (parsing==13).astype(np.uint8)
        palm_mask = self.get_palm(keypoints, parsing)
        retain_mask = shoes_mask + palm_mask + head_mask

        hand_leg_mask = (parsing==14).astype(np.uint8) + \
                        (parsing==15).astype(np.uint8) + \
                        (parsing==16).astype(np.uint8) + \
                        (parsing==17).astype(np.uint8)

        neck_mask = (parsing==10).astype(np.uint8)
        face_mask = (parsing==13).astype(np.uint8)

        skin_mask = neck_mask + face_mask
        skin = skin_mask * image
        skin_r = skin[..., 0].reshape((-1))
        skin_g = skin[..., 1].reshape((-1))
        skin_b = skin[..., 2].reshape((-1))
        skin_r_valid_index = np.where(skin_r > 0)[0]
        skin_g_valid_index = np.where(skin_g > 0)[0]
        skin_b_valid_index = np.where(skin_b > 0)[0]
        skin_r_median = np.median(
            skin_r[skin_r_valid_index]) * np.ones_like(image[...,0:1])
        skin_g_median = np.median(
            skin_g[skin_g_valid_index]) * np.ones_like(image[...,0:1])
        skin_b_median = np.median(
            skin_b[skin_b_valid_index]) * np.ones_like(image[...,0:1])
        skin_median = np.concatenate([skin_r_median, skin_g_median, skin_b_median], axis=2)

        tops_mask = (parsing==5).astype(np.uint8) + (parsing==7).astype(np.uint8)
        dresses_mask = (parsing==6).astype(np.uint8)
        lower_pants_mask = (parsing==9).astype(np.uint8)
        lower_skirt_mask = (parsing==12).astype(np.uint8)

        if np.sum(lower_pants_mask) > np.sum(lower_skirt_mask):
            lower_pants_mask += lower_skirt_mask
            lower_skirt_mask *= 0
        else:
            lower_skirt_mask += lower_pants_mask
            lower_pants_mask *= 0
       
        if np.sum(dresses_mask) > 0:
            if np.sum(lower_pants_mask) > 0:
                tops_mask += dresses_mask
                dresses_mask *= 0
            else:
                if np.sum(dresses_mask) > (np.sum(tops_mask)+np.sum(lower_skirt_mask)):
                    dresses_mask += (tops_mask + lower_skirt_mask)
                    tops_mask *= 0
                    lower_skirt_mask *= 0
                else:
                    if np.sum(tops_mask) > np.sum(lower_skirt_mask):
                        lower_skirt_mask += dresses_mask
                    else:
                        tops_mask += dresses_mask
                    dresses_mask *= 0

        gt_parsing = tops_mask * 1 + lower_pants_mask * 2 + lower_skirt_mask * 3 + \
                     dresses_mask * 4 + neck_mask * 5 + hand_leg_mask * 6

        lower_clothes_mask = lower_skirt_mask + lower_pants_mask
        upper_clothes_mask = tops_mask + dresses_mask
        upper_clothes_image = upper_clothes_mask * image
        lower_clothes_image = lower_clothes_mask * image

        upper_clothes_mask_rgb = np.concatenate([upper_clothes_mask,upper_clothes_mask,upper_clothes_mask],axis=2)
        lower_clothes_mask_rgb = np.concatenate([lower_clothes_mask,lower_clothes_mask,lower_clothes_mask],axis=2)
        upper_clothes_mask_rgb = upper_clothes_mask_rgb * 255
        lower_clothes_mask_rgb = lower_clothes_mask_rgb * 255

        lower_bbox = self.mask_to_bbox(lower_clothes_mask.copy())
        lower_clothes_upper_bound_for_train = np.zeros_like(lower_clothes_mask[...,0:1])
        if lower_bbox is not None:
            upper_bound = lower_bbox[1]
            lower_clothes_upper_bound_for_train[upper_bound:,...] += 255
        
        lower_clothes_upper_bound_for_test = np.zeros_like(lower_clothes_mask[...,0:1])
        left_hip_kps = keypoints[11]
        right_hip_kps = keypoints[8]
        if left_hip_kps[2] > 0.05 and right_hip_kps[2] > 0.05:
            hip_width = np.linalg.norm(left_hip_kps[0:2] - right_hip_kps[0:2])
            middle_hip_y = (left_hip_kps[1]+right_hip_kps[1]) / 2
            # upper_bound_via_kps = int(middle_hip_y - (hip_width / 3))
            upper_bound_via_kps = int(middle_hip_y - (hip_width / 2))
            if lower_bbox is not None:
                upper_bound = lower_bbox[1]
                if upper_bound_via_kps < upper_bound:
                    upper_bound = upper_bound_via_kps
            else:
                upper_bound = upper_bound_via_kps
            lower_clothes_upper_bound_for_test[upper_bound:,...] += 255
        elif lower_bbox is not None:
            upper_bound = lower_bbox[1]
            lower_clothes_upper_bound_for_test[upper_bound:,...] += 255

        norm_img, norm_img_lower, norm_img_lower_for_train, denorm_upper_img, denorm_lower_img, Ms, M_invs, \
                norm_clothes_masks, norm_clothes_masks_lower = self.normalize(upper_clothes_image, \
                lower_clothes_image, upper_clothes_mask_rgb, lower_clothes_mask_rgb, sleeve_mask, keypoints, 2)

        # ##### debug ####
        # cv2.imwrite('z_debug_image.png', image[...,[2,1,0]])
        # cv2.imwrite('z_debug_pants.png', (image*lower_pants_mask)[...,[2,1,0]])
        # cv2.imwrite('z_debug_skirts.png', (image*lower_skirt_mask)[...,[2,1,0]])
        # ###############

        lower_label_map = np.ones_like(lower_pants_mask)
        if np.sum(lower_pants_mask) > 0:
            lower_label_map *= 0
        elif np.sum(lower_skirt_mask) > 0:
            lower_label_map *= 1
        elif np.sum(dresses_mask) > 0:
            lower_label_map *= 2
        lower_label_map = lower_label_map / 2.0 * 255

        # ###### debug ####
        # pose_mask = (np.sum(pose,axis=2,keepdims=True)>0).astype(np.uint8)
        # image_vis = image * (1-pose_mask) + pose * pose_mask

        # cv2.imwrite('z_debug_image.png', image_vis[...,[2,1,0]])
        # lower_clothes_upper_bound_for_train_vis = np.concatenate([lower_clothes_upper_bound_for_train,lower_clothes_upper_bound_for_train,lower_clothes_upper_bound_for_train],axis=2)
        # lower_clothes_upper_bound_for_test_vis = np.concatenate([lower_clothes_upper_bound_for_test,lower_clothes_upper_bound_for_test,lower_clothes_upper_bound_for_test],axis=2)
        
        # lower_clothes_upper_bound_for_train_vis = lower_clothes_upper_bound_for_train_vis * (1-pose_mask) + pose * pose_mask
        # lower_clothes_upper_bound_for_test_vis = lower_clothes_upper_bound_for_test_vis * (1-pose_mask) + pose * pose_mask

        # lower_label_map_vis = np.concatenate([lower_label_map,lower_label_map,lower_label_map],axis=2)
        # cv2.imwrite('z_debug_upper_bound_train.png', lower_clothes_upper_bound_for_train_vis)
        # cv2.imwrite('z_debug_upper_bound_test.png', lower_clothes_upper_bound_for_test_vis)
        # cv2.imwrite('z_debug_lower_label.png', lower_label_map_vis)
        # for ii in range(lower_condition_for_train.shape[2]):
        #     condition_train_ii = lower_condition_for_train[...,ii:(ii+1)]
        #     condition_test_ii = lower_condition_for_test[...,ii:(ii+1)]
        #     condition_train_ii = np.concatenate([condition_train_ii,condition_train_ii,condition_train_ii],axis=2)
        #     condition_test_ii = np.concatenate([condition_test_ii,condition_test_ii,condition_test_ii],axis=2)
        #     cv2.imwrite('z_debug_condition_train_%d.png' % ii, condition_train_ii)
        #     cv2.imwrite('z_debug_condition_test_%d.png' % ii, condition_test_ii)
        # ################

        return image, pose, norm_img, norm_img_lower, norm_img_lower_for_train, denorm_upper_img, denorm_lower_img, Ms, \
                M_invs, gt_parsing, norm_clothes_masks, norm_clothes_masks_lower, retain_mask, skin_median, lower_label_map, \
                lower_clothes_upper_bound_for_train, lower_clothes_upper_bound_for_test


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if not os.path.exists(os.path.join(self._path, fname)):
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    ############################ get palm mask start #########################################

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps  = [x1,y1,x2,y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1<cos2:
            kps.extend([x3,y3,x4,y4])
        else:
            kps.extend([x4,y4,x3,y3])

        kps = np.array(kps).reshape(1,-1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask
    
    def get_hand_mask(self, hand_keypoints):
        # shoulder, elbow, wrist    
        s_x,s_y,s_c = hand_keypoints[0]
        e_x,e_y,e_c = hand_keypoints[1]
        w_x,w_y,w_c = hand_keypoints[2]

        h, w = 512, 512
        up_mask = np.ones((512,512,1),dtype=np.float32)
        bottom_mask = np.ones((512,512,1),dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            kernel = np.ones((35,35),np.uint8)
            up_mask = cv2.dilate(up_mask,kernel,iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            kernel = np.ones((28,28),np.uint8)
            bottom_mask = cv2.dilate(bottom_mask,kernel,iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[...,np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)==2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, keypoints, parsing):
        left_hand_keypoints = keypoints[[5,6,7],:].copy()
        right_hand_keypoints = keypoints[[2,3,4],:].copy()

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############################ get palm mask end #########################################

    def draw_pose_from_cords(self, pose_joints, img_size, radius=5, draw_joints=True):
        colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
        if draw_joints:
            for i, p in enumerate(limbseq):
                f, t = p[0]-1, p[1]-1
                from_missing = pose_joints[f][2] < 0.05
                to_missing = pose_joints[t][2] < 0.05

                if from_missing or to_missing:
                    continue

                pf = pose_joints[f][0], pose_joints[f][1]
                pt = pose_joints[t][0], pose_joints[t][1]
                fx, fy = pf[1], pf[0]# max(pf[1], 0), max(pf[0], 0)
                tx, ty = pt[1], pt[0]# max(pt[1], 0), max(pt[0], 0)
                fx, fy = int(fx), int(fy)# int(min(fx, 255)), int(min(fy, 191))
                tx, ty = int(tx), int(ty)# int(min(tx, 255)), int(min(ty, 191))
                cv2.line(colors, (fy, fx), (ty, tx), kptcolors[i], 5)

        for i, joint in enumerate(pose_joints):
            if pose_joints[i][2] < 0.05:
                continue
            if i == 9 or i == 10 or i == 12 or i == 13:
                if (pose_joints[i][0] <= 0) or \
                   (pose_joints[i][1] <= 0) or \
                   (pose_joints[i][0] >= img_size[1]-50) or \
                   (pose_joints[i][1] >= img_size[0]-50):
                    pose_joints[i][2] = 0.01
                    continue
            pj = joint[0], joint[1]
            x, y = int(pj[1]), int(pj[0])# int(min(pj[1], 255)), int(min(pj[0], 191))
            xx, yy = circle(x, y, radius=radius, shape=img_size)
            colors[xx, yy] = kptcolors[i]
        
        return colors, pose_joints

    def get_joints(self, keypoints_path):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            keypoints = np.zeros((18,3))
        else:
            keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        color_joint, keypoints = self.draw_pose_from_cords(keypoints, (512, 320))
        return color_joint, keypoints

    def valid_joints(self, joint):
        return (joint >= 0.1).all()

    def get_crop(self, keypoints, bpart, order, wh, o_w, o_h, ar = 1.0):
        joints = keypoints
        bpart_indices = [order.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices][:, :2])
        # fall backs
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            if bpart[0] == "lhip" and bpart[1] == "lknee":
                bpart = ["lhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":
                bpart = ["rhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lknee" and bpart[1] == 'lankle':
                bpart = ["lknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rknee" and bpart[1] == 'rankle':
                bpart = ["rknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])

        if not self.valid_joints(joints[bpart_indices][:, 2]):
                return None, None

        if part_src.shape[0] == 1:
            # leg fallback
            # hip_bpart = ["lhip", "rhip"]
            # hip_indices = [order.index(bb) for bb in hip_bpart]

            # if not self.valid_joints(joints[hip_indices][:,2]):
            #     return None, None
            # a = part_src[0]
            # # b = np.float32([a[0],o_h - 1])
            # part_hip = np.float32(joints[hip_indices][:,:2])
            # leg_height = 2 * np.linalg.norm(part_hip[0]-part_hip[1])
            # b = np.float32([a[0],a[1]+leg_height])
            # part_src = np.float32([a,b])

            torso_bpart = ["lhip", "rhip", "cneck"]
            torso_indices = [order.index(bb) for bb in torso_bpart]

            if not self.valid_joints(joints[torso_indices][:,2]):
                return None, None
            
            a = part_src[0]
            if 'lhip' in bpart:
                invalid_label = 'lknee'
            elif 'rhip' in bpart:
                invalid_label = 'rknee'
            elif 'lknee' in bpart:
                invalid_label = 'lankle'
            elif 'rknee' in bpart:
                invalid_label = 'rankle'
            invalid_joint = joints[order.index(invalid_label)]

            part_torso = np.float32(joints[torso_indices][:,:2])
            torso_length = np.linalg.norm(part_torso[2]-part_torso[1]) + \
                        np.linalg.norm(part_torso[2]-part_torso[0])
            torso_length = torso_length / 2

            if invalid_joint[2] > 0:
                direction = (invalid_joint[0:2]-a) / np.linalg.norm(a-invalid_joint[0:2])
                # if 'hip' in bpart[0]:
                #     b = a + torso_length * direction * 0.85
                # elif 'knee' in bpart[0]:
                #     b = a + torso_length * direction * 0.8
                # if 'hip' in bpart[0]:
                #     b = a + torso_length * direction * 0.90
                # elif 'knee' in bpart[0]:
                #     b = a + torso_length * direction * 0.85
                if 'hip' in bpart[0]:
                    b = a + torso_length * direction * 0.85
                elif 'knee' in bpart[0]:
                    b = a + torso_length * direction * 0.80
            else:
                # b = np.float32([a[0],a[1]+torso_length])
                if 'hip' in bpart[0]:
                    b = np.float32([a[0],a[1]+torso_length * 0.85])
                elif 'knee' in bpart[0]:
                    b = np.float32([a[0],a[1]+torso_length * 0.80])

            part_src = np.float32([a,b])

        if part_src.shape[0] == 4:
            hip_seg = (part_src[2] - part_src[1]) / 4
            hip_l = part_src[1]
            hip_r = part_src[2]
            hip_l_new = hip_l - hip_seg
            hip_r_new = hip_r + hip_seg
            if hip_l_new[0] > 0 and hip_l_new[1] > 0 and hip_l_new[0] < o_w and hip_l_new[1] < o_h:
                part_src[1] = hip_l_new
            if hip_r_new[0] > 0 and hip_r_new[1] > 0 and hip_r_new[0] < o_w and hip_r_new[1] < o_h:
                part_src[2] = hip_r_new

            shoulder_seg = (part_src[3] - part_src[0]) / 5
            shoulder_l = part_src[0]
            shoulder_r = part_src[3]
            shoulder_l_new = shoulder_l - shoulder_seg
            shoulder_r_new = shoulder_r + shoulder_seg
            if shoulder_l_new[0] > 0 and shoulder_l_new[1] > 0 and shoulder_l_new[0] < o_w and shoulder_l_new[1] < o_h:
                part_src[0] = shoulder_l_new
            if shoulder_r_new[0] > 0 and shoulder_r_new[1] > 0 and shoulder_r_new[0] < o_w and shoulder_r_new[1] < o_h:
                part_src[3] = shoulder_r_new
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            shoulder_seg = (part_src[0] - part_src[1]) / 5
            shoulder_l = part_src[1]
            shoulder_r = part_src[0]
            shoulder_l_new = shoulder_l - shoulder_seg
            shoulder_r_new = shoulder_r + shoulder_seg
            if shoulder_l_new[0] > 0 and shoulder_l_new[1] > 0 and shoulder_l_new[0] < o_w and shoulder_l_new[1] < o_h:
                part_src[1] = shoulder_l_new
            if shoulder_r_new[0] > 0 and shoulder_r_new[1] > 0 and shoulder_r_new[0] < o_w and shoulder_r_new[1] < o_h:
                part_src[0] = shoulder_r_new  

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            if normal[1] > 0.0:
                normal = -normal

            a = part_src[0] + normal
            b = part_src[0]
            c = part_src[1]
            d = part_src[1] + normal

            part_height = (c[1]+b[1])/2 - (a[1]+d[1])/2
            a[1] += part_height/2
            d[1] += part_height/2
            part_src = np.float32([d,c,b,a])
        else:
            assert part_src.shape[0] == 2                           
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            if 'rhip' in bpart or 'rknee' in bpart:
                # a = a + alpha*normal*1.5
                # d = d + alpha*normal*1.5
                a = a + alpha*normal*1.0
                d = d + alpha*normal*1.0
            if 'lhip' in bpart or 'lknee' in bpart:
                b = b - alpha*normal*1.0
                c = c - alpha*normal*1.0
            if 'relbow' in bpart or 'rwrist' in bpart:
                a = a + alpha*normal*0.45
                d = d + alpha*normal*0.45
                b = b - alpha*normal*0.1
                c = c - alpha*normal*0.1
            if 'lelbow' in bpart or 'lwrist' in bpart:
                a = a + alpha*normal*0.1
                d = d + alpha*normal*0.1
                b = b - alpha*normal*0.45
                c = c - alpha*normal*0.45
            part_src = np.float32([a,d,c,b])

        dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        M_inv = cv2.getPerspectiveTransform(part_dst,part_src)
        return M, M_inv

    def mask_to_bbox(self, mask):
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        site = np.where(mask > 0)
        if len(site[0]) > 0 and len(site[1]) > 0:
            bbox = [np.min(site[1]), np.min(site[0]),
                    np.max(site[1]), np.max(site[0])]
            return bbox
        else:
            return None

    def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask, \
                  sleeve_mask, keypoints, box_factor):

        h, w = upper_img.shape[:2]
        o_h, o_w = h, w
        h = h // 2**box_factor
        w = w // 2**box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
                ["rshoulder","rhip","lhip","lshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder","lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder","relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["lknee", "lankle"],
                ["rhip", "rknee"],
                ["rknee", "rankle"]]

        order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 
                'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee',  
                'lankle', 'reye', 'leye', 'rear', 'lear']
        # ar = 0.5

        part_imgs = list()
        part_imgs_lower = list()
        part_imgs_lower_for_train = list()
        part_clothes_masks = list()
        part_clothes_masks_lower = list()
        M_invs = list()
        Ms = list()

        denorm_upper_img = np.zeros_like(upper_img)
        denorm_lower_img = np.zeros_like(upper_img)
        kernel = np.ones((5,5),np.uint8)

        for ii, bpart in enumerate(bparts):
            if ii < 6:
                ar = 0.5
            else:
                ar = 0.4

            part_img = np.zeros((h, w, 3)).astype(np.uint8)
            part_img_lower = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask_lower = np.zeros((h,w,3)).astype(np.uint8)
            M, M_inv = self.get_crop(keypoints, bpart, order, wh, o_w, o_h, ar)

            if M is not None:
                if ii == 2 or ii == 3 or ii == 4 or ii == 5:
                    part_img = cv2.warpPerspective(upper_img*sleeve_mask, M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    part_clothes_mask = cv2.warpPerspective(upper_clothes_mask*sleeve_mask, M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                else:
                    part_img = cv2.warpPerspective(upper_img*(1-sleeve_mask), M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    part_clothes_mask = cv2.warpPerspective(upper_clothes_mask*(1-sleeve_mask), M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                
                denorm_patch = cv2.warpPerspective(part_img, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)[...,np.newaxis]
                denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)

                denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)

                if ii == 0 or ii >= 6:
                    part_img_lower = cv2.warpPerspective(lower_img, M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    part_clothes_mask_lower = cv2.warpPerspective(lower_clothes_mask, M, (w,h), borderMode = cv2.BORDER_CONSTANT)

                    denorm_patch_lower = cv2.warpPerspective(part_img_lower, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                    denorm_clothes_mask_patch_lower = cv2.warpPerspective(part_clothes_mask_lower, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                    denorm_clothes_mask_patch_lower = cv2.erode(denorm_clothes_mask_patch_lower, kernel, iterations=1)[...,np.newaxis]
                    denorm_clothes_mask_patch_lower = (denorm_clothes_mask_patch_lower==255).astype(np.uint8)

                    denorm_lower_img = denorm_patch_lower * denorm_clothes_mask_patch_lower + denorm_lower_img * (1-denorm_clothes_mask_patch_lower)

                Ms.append(M[np.newaxis,...])
                M_invs.append(M_inv[np.newaxis,...])
            else:
                Ms.append(np.zeros((1,3,3),dtype=np.float32))
                M_invs.append(np.zeros((1,3,3),dtype=np.float32))

            part_imgs.append(part_img)
            part_clothes_masks.append(part_clothes_mask)
            if ii == 0 or ii >= 6:
                part_imgs_lower.append(part_img_lower)
                part_imgs_lower_for_train.append(part_img_lower.copy())
                part_clothes_masks_lower.append(part_clothes_mask_lower)

        left_top_sleeve_mask = part_clothes_masks[2]
        right_top_sleeve_mask = part_clothes_masks[4]
        left_bottom_sleeve_mask = part_clothes_masks[3]
        right_bottom_sleeve_mask = part_clothes_masks[5]

        if np.sum(left_top_sleeve_mask) == 0 and np.sum(right_top_sleeve_mask) > 0:
            right_top_sleeve = part_imgs[4]
            left_top_sleeve = cv2.flip(right_top_sleeve,1)
            left_top_sleeve_mask = cv2.flip(right_top_sleeve_mask,1)
            part_imgs[2] = left_top_sleeve
            part_clothes_masks[2] = left_top_sleeve_mask
        elif np.sum(right_top_sleeve_mask) == 0 and np.sum(left_top_sleeve_mask) > 0:
            left_top_sleeve = part_imgs[2]
            right_top_sleeve = cv2.flip(left_top_sleeve,1)
            right_top_sleeve_mask = cv2.flip(left_top_sleeve_mask,1)
            part_imgs[4] = right_top_sleeve
            part_clothes_masks[4] = right_top_sleeve_mask

        if np.sum(left_bottom_sleeve_mask) == 0 and np.sum(right_bottom_sleeve_mask) > 0:
            right_bottom_sleeve = part_imgs[3]
            left_bottom_sleeve = cv2.flip(right_bottom_sleeve, 1)
            left_bottom_sleeve_mask = cv2.flip(right_bottom_sleeve_mask, 1)
            part_imgs[3] = left_bottom_sleeve
            part_clothes_masks[3] = left_bottom_sleeve_mask
        elif np.sum(right_bottom_sleeve_mask) == 0 and np.sum(left_bottom_sleeve_mask) > 0:
            left_bottom_sleeve = part_imgs[5]
            right_bottom_sleeve = cv2.flip(left_bottom_sleeve, 1)
            right_bottom_sleeve_mask = cv2.flip(left_bottom_sleeve_mask, 1)
            part_imgs[5] = right_bottom_sleeve
            part_clothes_masks[5] = right_bottom_sleeve_mask


        # ###### debug ######
        # # cv2.imwrite('z_debug_upper.png', upper_img[...,[2,1,0]])
        # # for ii, p in enumerate(part_imgs):
        # #     # if ii > 5:
        # #     #     break
        # #     cv2.imwrite('z_debug_part_%d_new.png' % ii , p[...,[2,1,0]])
        # # cv2.imwrite('z_debug_denorm_upper.png', denorm_upper_img[...,[2,1,0]])
        # cv2.imwrite('z_debug_lower.png', lower_img[...,[2,1,0]])
        # for ii, p in enumerate(part_imgs_lower):
        #     if ii == 0 or ii == 1 or ii == 3:
        #         cv2.imwrite('z_debug_part_%d_lower.png' % ii, p[...,[2,1,0]])
        # cv2.imwrite('z_debug_denorm_lower.png', denorm_lower_img[...,[2,1,0]])
        # ################

        part_cloth_mask_lower = part_clothes_masks_lower[0][...,0:1]
        bbox_lower = self.mask_to_bbox(part_cloth_mask_lower.copy())
        if bbox_lower is not None:
            # if random.random()<0.6:
            #     if random.random()<0.5:
            #         part_imgs_lower_for_train[0] = np.zeros((h,w,3)).astype(np.uint8)
            #         if random.random() < 0.5:
            #             erase_length = random.randint(1,(h//10))
            #             part_imgs_lower_for_train[1][0:erase_length,...] *= 0
            #             part_imgs_lower_for_train[3][0:erase_length,...] *= 0
            #     else:
            #         ty= bbox_lower[1]
            #         by = random.randint(ty+1,h-1)
            #         part_imgs_lower_for_train[0][ty:by,...] *= 0
            if random.random()<0.80:
                if random.random()<0.6:
                    part_imgs_lower_for_train[0] = np.zeros((h,w,3)).astype(np.uint8)
                    if random.random() < 0.75:
                        erase_length = random.randint(1,(h//10))
                        part_imgs_lower_for_train[1][0:erase_length,...] *= 0
                        part_imgs_lower_for_train[3][0:erase_length,...] *= 0
                else:
                    ty= bbox_lower[1]
                    by = random.randint(ty+1,h)
                    part_imgs_lower_for_train[0][ty:by,...] *= 0

        ######## debug
        # cv2.imwrite('z_debug_lower.png', lower_img[...,[2,1,0]])
        # cv2.imwrite('z_debug_lower_mask.png', lower_clothes_mask[...,[2,1,0]])
        # for ii, p in enumerate(part_imgs_lower_for_train):
        #     if ii == 0 or ii == 1 or ii == 3:
        #         cv2.imwrite('z_debug_part_%d_lower_erase.png' % ii, p[...,[2,1,0]])
        # for ii, p in enumerate(part_imgs_lower):
        #     if ii == 0 or ii == 1 or ii == 3:
        #         cv2.imwrite('z_debug_part_%d_lower.png' % ii, p[...,[2,1,0]])
        # for ii, p in enumerate(part_clothes_masks_lower):
        #     if ii == 0 or ii == 1 or ii == 3:
        #         cv2.imwrite('z_debug_part_%d_lower_mask.png' % ii, p[...,[2,1,0]])
        ########

        img = np.concatenate(part_imgs, axis = 2)
        img_lower = np.concatenate(part_imgs_lower, axis=2)
        img_lower_for_train = np.concatenate(part_imgs_lower_for_train,axis=2)
        clothes_masks = np.concatenate(part_clothes_masks, axis=2)
        clothes_masks_lower = np.concatenate(part_clothes_masks_lower, axis=2)
        M_invs = np.concatenate(M_invs, axis=0)
        Ms = np.concatenate(Ms, axis=0)

        return img, img_lower, img_lower_for_train, denorm_upper_img, denorm_lower_img, Ms, \
               M_invs, clothes_masks, clothes_masks_lower


    def __getitem__(self, idx):
        image, pose, norm_img, norm_img_lower, norm_img_lower_for_train, denorm_upper_img, denorm_lower_img, Ms, \
            M_invs, gt_parsing, norm_clothes_masks, norm_clothes_masks_lower, retain_mask, skin_median, lower_label_map, \
            lower_clothes_upper_bound_for_train, lower_clothes_upper_bound_for_test = self._load_raw_image(self._raw_idx[idx])

        image = image.transpose(2, 0, 1)                    # HWC => CHW
        pose = pose.transpose(2, 0, 1)                      # HWC => CHW
        norm_img = norm_img.transpose(2, 0, 1)
        norm_img_lower = norm_img_lower.transpose(2,0,1)
        norm_img_lower_for_train = norm_img_lower_for_train.transpose(2,0,1)
        denorm_upper_img = denorm_upper_img.transpose(2,0,1)
        denorm_lower_img = denorm_lower_img.transpose(2,0,1)

        norm_clothes_masks = norm_clothes_masks.transpose(2,0,1)
        norm_clothes_masks_lower = norm_clothes_masks_lower.transpose(2,0,1)

        gt_parsing = gt_parsing.transpose(2,0,1)

        retain_mask = retain_mask.transpose(2,0,1)
        skin_median = skin_median.transpose(2,0,1)
        lower_label_map = lower_label_map.transpose(2,0,1)

        lower_clothes_upper_bound_for_train = lower_clothes_upper_bound_for_train.transpose(2,0,1)
        lower_clothes_upper_bound_for_test = lower_clothes_upper_bound_for_test.transpose(2,0,1)

        denorm_random_mask = np.zeros((512,512,1),dtype=np.uint8)
        denorm_random_mask_bottom = np.zeros((512,512,1),dtype=np.uint8)

        if random.random() < 0.9:
            fname = self._random_mask_acgpn_fnames[self._raw_idx[idx]%self._mask_acgpn_numbers]
            random_mask = cv2.imread(fname)[...,0:1]
            denorm_random_mask += random_mask
            denorm_random_mask_bottom += random_mask

        denorm_random_mask = (denorm_random_mask>0).astype(np.uint8)
        denorm_random_mask = denorm_random_mask.transpose(2,0,1)

        denorm_random_mask_bottom = (denorm_random_mask_bottom>0).astype(np.uint8)
        denorm_random_mask_bottom = denorm_random_mask_bottom.transpose(2,0,1)

        denorm_upper_img_erase = denorm_upper_img * (1-denorm_random_mask)
        denorm_upper_mask = (np.sum(denorm_upper_img_erase, axis=0, keepdims=True)>0).astype(np.uint8)
        denorm_lower_img_erase = denorm_lower_img * (1-denorm_random_mask_bottom)
        denorm_lower_mask = (np.sum(denorm_lower_img_erase, axis=0, keepdims=True)>0).astype(np.uint8)

        return image.copy(), pose.copy(), norm_img.copy(), norm_img_lower.copy(), norm_img_lower_for_train.copy(), \
                denorm_upper_img_erase.copy(), denorm_lower_img_erase.copy(), Ms.copy(), \
                M_invs.copy(), gt_parsing.copy(), denorm_upper_mask.copy(), denorm_lower_mask.copy(), \
                norm_clothes_masks.copy(), norm_clothes_masks_lower.copy(), retain_mask.copy(), \
                skin_median.copy(), lower_label_map.copy(), lower_clothes_upper_bound_for_train.copy(), \
                lower_clothes_upper_bound_for_test.copy()


class UvitonDatasetFull_512_test_full(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        test_txt,
        use_sleeve_mask,
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.use_sleeve_mask = use_sleeve_mask

        if os.path.isdir(self._path):
            self._type = 'dir' 
            self._image_fnames = []
            self._kpt_fnames = []
            self._parsing_fnames = []

            self._clothes_image_fnames = []
            self._clothes_kpt_fnames = []
            self._clothes_parsing_fnames = []
            self._clothes_garment_parsing_fnames = []

            txt_path = os.path.join(self._path, test_txt)
            with open(txt_path, 'r') as f:
                for ii, line in enumerate(f.readlines()):
                    clothes_name, person_name = line.strip().split()

                    self._image_fnames.append(os.path.join('image', person_name))
                    self._kpt_fnames.append(os.path.join('keypoints', person_name[:-4]+'_keypoints.json'))
                    self._parsing_fnames.append(os.path.join('parsing',person_name.replace('.jpg','.png')))

                    self._clothes_image_fnames.append(os.path.join('image', clothes_name))
                    self._clothes_kpt_fnames.append(os.path.join('keypoints', clothes_name[:-4]+'_keypoints.json'))
                    self._clothes_garment_parsing_fnames.append(os.path.join('garment_parsing', clothes_name[:-4]+'.png'))
                    self._clothes_parsing_fnames.append(os.path.join('parsing',clothes_name.replace('.jpg','.png')))

        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        im_shape = list((self._load_raw_image(0))[0].shape)
        raw_shape = [len(self._image_fnames)] + [im_shape[2], im_shape[0], im_shape[1]]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        # load images --> range [0, 255]
        fname = self._image_fnames[raw_idx]
        person_name = fname
        f = os.path.join(self._path, fname)
        self.image = np.array(PIL.Image.open(f))
        im_shape = self.image.shape
        h, w = im_shape[0], im_shape[1]
        left_padding = (h-w) // 2
        right_padding = h-w-left_padding
        image = np.pad(self.image,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        # load keypoints --> range [0, 1]
        fname = self._kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        pose, keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        pose = np.pad(pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        keypoints[:,0] += left_padding

        # load upper_cloth and lower body
        fname = self._parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        parsing = cv2.imread(f)[...,0:1]
        parsing = np.pad(parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))

        shoes_mask = (parsing==18).astype(np.uint8) + (parsing==19).astype(np.uint8)
        head_mask = (parsing==1).astype(np.uint8) + (parsing==2).astype(np.uint8) + \
                    (parsing==4).astype(np.uint8) + (parsing==13).astype(np.uint8)
        palm_mask = self.get_palm(keypoints, parsing)
        retain_mask = shoes_mask + palm_mask + head_mask

        neck_mask = (parsing==10).astype(np.uint8)
        face_mask = (parsing==13).astype(np.uint8)
        skin_mask = neck_mask + face_mask
        skin = skin_mask * image
        skin_r = skin[..., 0].reshape((-1))
        skin_g = skin[..., 1].reshape((-1))
        skin_b = skin[..., 2].reshape((-1))
        skin_r_valid_index = np.where(skin_r > 0)[0]
        skin_g_valid_index = np.where(skin_g > 0)[0]
        skin_b_valid_index = np.where(skin_b > 0)[0]
        skin_r_median = np.median(
            skin_r[skin_r_valid_index]) * np.ones_like(image[...,0:1])
        skin_g_median = np.median(
            skin_g[skin_g_valid_index]) * np.ones_like(image[...,0:1])
        skin_b_median = np.median(
            skin_b[skin_b_valid_index]) * np.ones_like(image[...,0:1])
        skin_average = np.concatenate([skin_r_median, skin_g_median, skin_b_median], axis=2)

        ##### clothes items
        fname = self._clothes_image_fnames[raw_idx]
        clothes_name = fname
        f = os.path.join(self._path, fname)
        self.clothes = np.array(PIL.Image.open(f))
        clothes = np.pad(self.clothes,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        fname = self._clothes_kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        clothes_pose, clothes_keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        clothes_pose = np.pad(clothes_pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        clothes_keypoints[:,0] += left_padding

        fname = self._clothes_parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        clothes_parsing = cv2.imread(f)[...,0:1]
        clothes_parsing = np.pad(clothes_parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))

        tops_mask = (clothes_parsing==5).astype(np.uint8) + (clothes_parsing==7).astype(np.uint8)
        dresses_mask = (clothes_parsing==6).astype(np.uint8)
        lower_pants_mask = (clothes_parsing==9).astype(np.uint8)
        lower_skirt_mask = (clothes_parsing==12).astype(np.uint8)

        if np.sum(lower_pants_mask) > np.sum(lower_skirt_mask):
            lower_pants_mask += lower_skirt_mask
            lower_skirt_mask *= 0
        else:
            lower_skirt_mask += lower_pants_mask
            lower_pants_mask *= 0
       
        if np.sum(dresses_mask) > 0:
            if np.sum(lower_pants_mask) > 0:
                tops_mask += dresses_mask
                dresses_mask *= 0
            else:
                if np.sum(dresses_mask) > (np.sum(tops_mask)+np.sum(lower_skirt_mask)):
                    dresses_mask += (tops_mask + lower_skirt_mask)
                    tops_mask *= 0
                    lower_skirt_mask *= 0
                else:
                    if np.sum(tops_mask) > np.sum(lower_skirt_mask):
                        lower_skirt_mask += dresses_mask
                    else:
                        tops_mask += dresses_mask
                    dresses_mask *= 0

        upper_clothes_mask = tops_mask + dresses_mask
        lower_clothes_mask = lower_skirt_mask + lower_pants_mask

        upper_clothes_image = upper_clothes_mask * clothes
        lower_clothes_image = lower_clothes_mask * clothes

        upper_clothes_mask_rgb = np.concatenate([upper_clothes_mask,upper_clothes_mask,upper_clothes_mask],axis=2)
        lower_clothes_mask_rgb = np.concatenate([lower_clothes_mask,lower_clothes_mask,lower_clothes_mask],axis=2)
        upper_clothes_mask_rgb = upper_clothes_mask_rgb * 255
        lower_clothes_mask_rgb = lower_clothes_mask_rgb * 255
    
        sleeve_mask = None
        if self.use_sleeve_mask:
            fname = self._clothes_garment_parsing_fnames[raw_idx]
            f = os.path.join(self._path, fname)
            garment_parsing = cv2.imread(f)[...,0:1]
            garment_parsing = np.pad(garment_parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))
            sleeve_mask = (garment_parsing==10).astype(np.uint8) + (garment_parsing==11).astype(np.uint8)

        norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img = self.normalize(upper_clothes_image, lower_clothes_image, \
                upper_clothes_mask_rgb, lower_clothes_mask_rgb, sleeve_mask, clothes_keypoints, keypoints, 2)

        denorm_lower_img_mask = (np.sum(denorm_lower_img, axis=2, keepdims=True)>0).astype(np.uint8)
        lower_clothes_upper_bound = np.zeros_like(lower_clothes_mask)
        lower_bbox = self.mask_to_bbox(denorm_lower_img_mask)
        if lower_bbox is not None:
            upper_bound = lower_bbox[1]
            lower_clothes_upper_bound[upper_bound:,...] += 255

        lower_label_map = np.ones_like(lower_clothes_mask)
        if np.sum(lower_pants_mask) > 0:
            lower_label_map *= 0
        elif np.sum(lower_skirt_mask) > 0:
            lower_label_map *= 1
        elif np.sum(dresses_mask) > 0:
            lower_label_map *= 2
            lower_clothes_upper_bound *= 0
        lower_label_map = lower_label_map / 2.0 * 255

        return image, clothes, pose, clothes_pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, \
                retain_mask, skin_average, lower_label_map, lower_clothes_upper_bound, person_name, clothes_name


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if not os.path.exists(os.path.join(self._path, fname)):
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    
    ############################ get palm mask start #########################################

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps  = [x1,y1,x2,y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1<cos2:
            kps.extend([x3,y3,x4,y4])
        else:
            kps.extend([x4,y4,x3,y3])

        kps = np.array(kps).reshape(1,-1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask
    
    def get_hand_mask(self, hand_keypoints):
        # shoulder, elbow, wrist    
        s_x,s_y,s_c = hand_keypoints[0]
        e_x,e_y,e_c = hand_keypoints[1]
        w_x,w_y,w_c = hand_keypoints[2]

        h, w = 512, 512
        up_mask = np.ones((512,512,1),dtype=np.float32)
        bottom_mask = np.ones((512,512,1),dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            kernel = np.ones((35,35),np.uint8)
            up_mask = cv2.dilate(up_mask,kernel,iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            kernel = np.ones((28,28),np.uint8)
            bottom_mask = cv2.dilate(bottom_mask,kernel,iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[...,np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)==2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, keypoints, parsing):
        left_hand_keypoints = keypoints[[5,6,7],:].copy()
        right_hand_keypoints = keypoints[[2,3,4],:].copy()

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############################ get palm mask end #########################################

    def draw_pose_from_cords(self, pose_joints, img_size, radius=5, draw_joints=True):
        colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
        if draw_joints:
            for i, p in enumerate(limbseq):
                f, t = p[0]-1, p[1]-1
                from_missing = pose_joints[f][2] < 0.05
                to_missing = pose_joints[t][2] < 0.05

                if from_missing or to_missing:
                    continue

                pf = pose_joints[f][0], pose_joints[f][1]
                pt = pose_joints[t][0], pose_joints[t][1]
                fx, fy = pf[1], pf[0]# max(pf[1], 0), max(pf[0], 0)
                tx, ty = pt[1], pt[0]# max(pt[1], 0), max(pt[0], 0)
                fx, fy = int(fx), int(fy)# int(min(fx, 255)), int(min(fy, 191))
                tx, ty = int(tx), int(ty)# int(min(tx, 255)), int(min(ty, 191))
                cv2.line(colors, (fy, fx), (ty, tx), kptcolors[i], 5)

        for i, joint in enumerate(pose_joints):
            if pose_joints[i][2] < 0.05:
                continue
            if i == 9 or i == 10 or i == 12 or i == 13:
                if (pose_joints[i][0] <= 0) or \
                   (pose_joints[i][1] <= 0) or \
                   (pose_joints[i][0] >= img_size[1]-50) or \
                   (pose_joints[i][1] >= img_size[0]-50):
                    pose_joints[i][2] = 0.01
                    continue
            pj = joint[0], joint[1]
            x, y = int(pj[1]), int(pj[0])# int(min(pj[1], 255)), int(min(pj[0], 191))
            xx, yy = circle(x, y, radius=radius, shape=img_size)
            colors[xx, yy] = kptcolors[i]
        
        return colors, pose_joints

    def get_joints(self, keypoints_path):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            keypoints = np.zeros((18,3))
        else:
            keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        color_joint, keypoints = self.draw_pose_from_cords(keypoints, (512, 320))
        return color_joint, keypoints

    def valid_joints(self, joint):
        return (joint >= 0.1).all()

    def get_crop(self, keypoints, bpart, order, wh, o_w, o_h, ar = 1.0):
        joints = keypoints
        bpart_indices = [order.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices][:, :2])
        # fall backs
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            if bpart[0] == "lhip" and bpart[1] == "lknee":
                bpart = ["lhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":
                bpart = ["rhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lknee" and bpart[1] == 'lankle':
                bpart = ["lknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rknee" and bpart[1] == 'rankle':
                bpart = ["rknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])

        if not self.valid_joints(joints[bpart_indices][:, 2]):
                return None, None

        if part_src.shape[0] == 1:
            # leg fallback
            # hip_bpart = ["lhip", "rhip"]
            # hip_indices = [order.index(bb) for bb in hip_bpart]

            # if not self.valid_joints(joints[hip_indices][:,2]):
            #     return None, None
            # a = part_src[0]
            # # b = np.float32([a[0],o_h - 1])
            # part_hip = np.float32(joints[hip_indices][:,:2])
            # leg_height = 2 * np.linalg.norm(part_hip[0]-part_hip[1])
            # b = np.float32([a[0],a[1]+leg_height])
            # part_src = np.float32([a,b])

            torso_bpart = ["lhip", "rhip", "cneck"]
            torso_indices = [order.index(bb) for bb in torso_bpart]

            if not self.valid_joints(joints[torso_indices][:,2]):
                return None, None
            
            a = part_src[0]
            if 'lhip' in bpart:
                invalid_label = 'lknee'
            elif 'rhip' in bpart:
                invalid_label = 'rknee'
            elif 'lknee' in bpart:
                invalid_label = 'lankle'
            elif 'rknee' in bpart:
                invalid_label = 'rankle'
            invalid_joint = joints[order.index(invalid_label)]

            part_torso = np.float32(joints[torso_indices][:,:2])
            torso_length = np.linalg.norm(part_torso[2]-part_torso[1]) + \
                        np.linalg.norm(part_torso[2]-part_torso[0])
            torso_length = torso_length / 2

            if invalid_joint[2] > 0:
                direction = (invalid_joint[0:2]-a) / np.linalg.norm(a-invalid_joint[0:2])
                # if 'hip' in bpart[0]:
                #     b = a + torso_length * direction * 0.85
                # elif 'knee' in bpart[0]:
                #     b = a + torso_length * direction * 0.8
                # if 'hip' in bpart[0]:
                #     b = a + torso_length * direction * 0.90
                # elif 'knee' in bpart[0]:
                #     b = a + torso_length * direction * 0.85
                if 'hip' in bpart[0]:
                    b = a + torso_length * direction * 0.85
                elif 'knee' in bpart[0]:
                    b = a + torso_length * direction * 0.80
            else:
                # b = np.float32([a[0],a[1]+torso_length])
                if 'hip' in bpart[0]:
                    b = np.float32([a[0],a[1]+torso_length * 0.85])
                elif 'knee' in bpart[0]:
                    b = np.float32([a[0],a[1]+torso_length * 0.80])

            part_src = np.float32([a,b])

        if part_src.shape[0] == 4:
            hip_seg = (part_src[2] - part_src[1]) / 4
            hip_l = part_src[1]
            hip_r = part_src[2]
            hip_l_new = hip_l - hip_seg
            hip_r_new = hip_r + hip_seg
            if hip_l_new[0] > 0 and hip_l_new[1] > 0 and hip_l_new[0] < o_w and hip_l_new[1] < o_h:
                part_src[1] = hip_l_new
            if hip_r_new[0] > 0 and hip_r_new[1] > 0 and hip_r_new[0] < o_w and hip_r_new[1] < o_h:
                part_src[2] = hip_r_new

            shoulder_seg = (part_src[3] - part_src[0]) / 5
            shoulder_l = part_src[0]
            shoulder_r = part_src[3]
            shoulder_l_new = shoulder_l - shoulder_seg
            shoulder_r_new = shoulder_r + shoulder_seg
            if shoulder_l_new[0] > 0 and shoulder_l_new[1] > 0 and shoulder_l_new[0] < o_w and shoulder_l_new[1] < o_h:
                part_src[0] = shoulder_l_new
            if shoulder_r_new[0] > 0 and shoulder_r_new[1] > 0 and shoulder_r_new[0] < o_w and shoulder_r_new[1] < o_h:
                part_src[3] = shoulder_r_new
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            shoulder_seg = (part_src[0] - part_src[1]) / 5
            shoulder_l = part_src[1]
            shoulder_r = part_src[0]
            shoulder_l_new = shoulder_l - shoulder_seg
            shoulder_r_new = shoulder_r + shoulder_seg
            if shoulder_l_new[0] > 0 and shoulder_l_new[1] > 0 and shoulder_l_new[0] < o_w and shoulder_l_new[1] < o_h:
                part_src[1] = shoulder_l_new
            if shoulder_r_new[0] > 0 and shoulder_r_new[1] > 0 and shoulder_r_new[0] < o_w and shoulder_r_new[1] < o_h:
                part_src[0] = shoulder_r_new  

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            if normal[1] > 0.0:
                normal = -normal

            a = part_src[0] + normal
            b = part_src[0]
            c = part_src[1]
            d = part_src[1] + normal

            part_height = (c[1]+b[1])/2 - (a[1]+d[1])/2
            a[1] += part_height/2
            d[1] += part_height/2
            part_src = np.float32([d,c,b,a])
        else:
            assert part_src.shape[0] == 2                           
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            if 'rhip' in bpart or 'rknee' in bpart:
                # a = a + alpha*normal*1.5
                # d = d + alpha*normal*1.5
                a = a + alpha*normal*1.0
                d = d + alpha*normal*1.0
            if 'lhip' in bpart or 'lknee' in bpart:
                b = b - alpha*normal*1.0
                c = c - alpha*normal*1.0
            if 'relbow' in bpart or 'rwrist' in bpart:
                a = a + alpha*normal*0.45
                d = d + alpha*normal*0.45
                b = b - alpha*normal*0.1
                c = c - alpha*normal*0.1
            if 'lelbow' in bpart or 'lwrist' in bpart:
                a = a + alpha*normal*0.1
                d = d + alpha*normal*0.1
                b = b - alpha*normal*0.45
                c = c - alpha*normal*0.45
            part_src = np.float32([a,d,c,b])

        dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        M_inv = cv2.getPerspectiveTransform(part_dst,part_src)
        return M, M_inv

    def mask_to_bbox(self, mask):
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        site = np.where(mask > 0)
        if len(site[0]) > 0 and len(site[1]) > 0:
            bbox = [np.min(site[1]), np.min(site[0]),
                    np.max(site[1]), np.max(site[0])]
            return bbox
        else:
            return None

    def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask, \
                    sleeve_mask, clothes_keypoints, person_keypoints, box_factor):
        h, w = upper_img.shape[:2]
        o_h, o_w = h, w
        h = h // 2**box_factor
        w = w // 2**box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
                ["rshoulder","rhip","lhip","lshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder","lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder","relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["lknee", "lankle"],
                ["rhip", "rknee"],
                ["rknee", "rankle"]]
        order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 
                'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee',  
                'lankle', 'reye', 'leye', 'rear', 'lear']

        part_imgs = list()
        part_imgs_lower = list()
        part_clothes_masks = list()
        part_clothes_masks_lower = list()

        denorm_upper_img = np.zeros_like(upper_img)
        denorm_lower_img = np.zeros_like(upper_img)
        kernel = np.ones((5,5),np.uint8)

        for ii, bpart in enumerate(bparts):
            if ii < 6:
                ar = 0.5
            else:
                ar = 0.4

            part_img = np.zeros((h, w, 3)).astype(np.uint8)
            part_img_lower = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask_lower = np.zeros((h,w,3)).astype(np.uint8)
            
            clothes_M, _ = self.get_crop(clothes_keypoints, bpart, order, wh, o_w, o_h, ar)
            person_M, person_M_inv = self.get_crop(person_keypoints, bpart, order, wh, o_w, o_h, ar)

            if clothes_M is not None:
                if ii == 2 or ii == 3 or ii == 4 or ii == 5:
                    if sleeve_mask is not None:
                        part_img = cv2.warpPerspective(upper_img*sleeve_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask*sleeve_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    else:
                        part_img = cv2.warpPerspective(upper_img, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)                  
                else:
                    if sleeve_mask is not None:
                        part_img = cv2.warpPerspective(upper_img*(1-sleeve_mask), clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask*(1-sleeve_mask), clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    else:
                        part_img = cv2.warpPerspective(upper_img, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                if person_M_inv is not None:
                    denorm_patch = cv2.warpPerspective(part_img, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                    # part_img = cv2.warpPerspective(denorm_patch, person_M, (w,h), borderMode=cv2.BORDER_CONSTANT)

                    denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                    denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)[...,np.newaxis]
                    denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)

                    denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)

            if ii == 0 or ii >= 6:
                if clothes_M is not None:
                    part_img_lower = cv2.warpPerspective(lower_img, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    part_clothes_mask_lower = cv2.warpPerspective(lower_clothes_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)

                    if person_M_inv is not None:
                        denorm_patch_lower = cv2.warpPerspective(part_img_lower, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)                        
                        # part_img_lower = cv2.warpPerspective(denorm_patch_lower, person_M, (w,h), borderMode=cv2.BORDER_CONSTANT)
                        
                        denorm_clothes_mask_patch_lower = cv2.warpPerspective(part_clothes_mask_lower, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                        denorm_clothes_mask_patch_lower = cv2.erode(denorm_clothes_mask_patch_lower, kernel, iterations=1)[...,np.newaxis]
                        denorm_clothes_mask_patch_lower = (denorm_clothes_mask_patch_lower==255).astype(np.uint8)

                        denorm_lower_img = denorm_patch_lower * denorm_clothes_mask_patch_lower + denorm_lower_img * (1-denorm_clothes_mask_patch_lower)

            part_imgs.append(part_img)
            part_clothes_masks.append(part_clothes_mask)
            if ii == 0 or ii >= 6:
                part_imgs_lower.append(part_img_lower)
                part_clothes_masks_lower.append(part_clothes_mask_lower)

        left_top_sleeve_mask = part_clothes_masks[2]
        right_top_sleeve_mask = part_clothes_masks[4]
        left_bottom_sleeve_mask = part_clothes_masks[3]
        right_bottom_sleeve_mask = part_clothes_masks[5]

        if np.sum(left_top_sleeve_mask) == 0 and np.sum(right_top_sleeve_mask) > 0:
            right_top_sleeve = part_imgs[4]
            left_top_sleeve = cv2.flip(right_top_sleeve,1)
            left_top_sleeve_mask = cv2.flip(right_top_sleeve_mask,1)
            part_imgs[2] = left_top_sleeve
            part_clothes_masks[2] = left_top_sleeve_mask
        elif np.sum(right_top_sleeve_mask) == 0 and np.sum(left_top_sleeve_mask) > 0:
            left_top_sleeve = part_imgs[2]
            right_top_sleeve = cv2.flip(left_top_sleeve,1)
            right_top_sleeve_mask = cv2.flip(left_top_sleeve_mask,1)
            part_imgs[4] = right_top_sleeve
            part_clothes_masks[4] = right_top_sleeve_mask

        if np.sum(left_bottom_sleeve_mask) == 0 and np.sum(right_bottom_sleeve_mask) > 0:
            right_bottom_sleeve = part_imgs[3]
            left_bottom_sleeve = cv2.flip(right_bottom_sleeve, 1)
            left_bottom_sleeve_mask = cv2.flip(right_bottom_sleeve_mask, 1)
            part_imgs[3] = left_bottom_sleeve
            part_clothes_masks[3] = left_bottom_sleeve_mask
        elif np.sum(right_bottom_sleeve_mask) == 0 and np.sum(left_bottom_sleeve_mask) > 0:
            left_bottom_sleeve = part_imgs[5]
            right_bottom_sleeve = cv2.flip(left_bottom_sleeve, 1)
            right_bottom_sleeve_mask = cv2.flip(left_bottom_sleeve_mask, 1)
            part_imgs[5] = right_bottom_sleeve
            part_clothes_masks[5] = right_bottom_sleeve_mask

        img = np.concatenate(part_imgs, axis = 2)
        img_lower = np.concatenate(part_imgs_lower, axis=2)

        return img, img_lower, denorm_upper_img, denorm_lower_img

    def __getitem__(self, idx):
        image, clothes, pose, clothes_pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, \
            retain_mask, skin_average, lower_label_map, lower_clothes_upper_bound, \
            person_name, clothes_name = self._load_raw_image(self._raw_idx[idx])

        image = image.transpose(2, 0, 1)                    # HWC => CHW
        clothes = clothes.transpose(2,0,1)
        pose = pose.transpose(2, 0, 1)                      # HWC => CHW
        clothes_pose = clothes_pose.transpose(2,0,1)
        norm_img = norm_img.transpose(2, 0, 1)
        norm_img_lower = norm_img_lower.transpose(2,0,1)
        denorm_upper_img = denorm_upper_img.transpose(2,0,1)
        denorm_lower_img = denorm_lower_img.transpose(2,0,1)
        denorm_upper_mask = (np.sum(denorm_upper_img, axis=0, keepdims=True)>0).astype(np.uint8)
        denorm_lower_mask = (np.sum(denorm_lower_img, axis=0, keepdims=True)>0).astype(np.uint8)

        skin_average = skin_average.transpose(2,0,1)
        retain_mask = retain_mask.transpose(2,0,1)
        lower_label_map = lower_label_map.transpose(2,0,1)
        lower_clothes_upper_bound = lower_clothes_upper_bound.transpose(2,0,1)

        return image.copy(), clothes.copy(), pose.copy(), clothes_pose.copy(), norm_img.copy(), norm_img_lower.copy(), \
               denorm_upper_img.copy(), denorm_lower_img.copy(), denorm_upper_mask.copy(), denorm_lower_mask.copy(), \
               retain_mask.copy(), skin_average.copy(), lower_label_map.copy(), lower_clothes_upper_bound.copy(),\
               person_name, clothes_name


class UvitonDatasetFull_512_test_upper(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        test_txt,
        use_sleeve_mask,
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.use_sleeve_mask = use_sleeve_mask

        if os.path.isdir(self._path):
            self._type = 'dir' 
            self._image_fnames = []
            self._kpt_fnames = []
            self._parsing_fnames = []

            self._clothes_image_fnames = []
            self._clothes_kpt_fnames = []
            self._clothes_parsing_fnames = []
            self._clothes_garment_parsing_fnames = []

            txt_path = os.path.join(self._path, test_txt)
            with open(txt_path, 'r') as f:
                for ii, line in enumerate(f.readlines()):
                    clothes_name, person_name = line.strip().split()

                    self._image_fnames.append(os.path.join('image', person_name))
                    self._kpt_fnames.append(os.path.join('keypoints', person_name[:-4]+'_keypoints.json'))
                    self._parsing_fnames.append(os.path.join('parsing',person_name.replace('.jpg','.png')))

                    self._clothes_image_fnames.append(os.path.join('image', clothes_name))
                    self._clothes_kpt_fnames.append(os.path.join('keypoints', clothes_name[:-4]+'_keypoints.json'))
                    self._clothes_garment_parsing_fnames.append(os.path.join('garment_parsing', clothes_name[:-4]+'.png'))
                    self._clothes_parsing_fnames.append(os.path.join('parsing',clothes_name.replace('.jpg','.png')))
            
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        im_shape = list((self._load_raw_image(0))[0].shape)
        raw_shape = [len(self._image_fnames)] + [im_shape[2], im_shape[0], im_shape[1]]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        # load images --> range [0, 255]
        fname = self._image_fnames[raw_idx]
        person_name = fname
        f = os.path.join(self._path, fname)
        self.image = np.array(PIL.Image.open(f))
        im_shape = self.image.shape
        h, w = im_shape[0], im_shape[1]
        left_padding = (h-w) // 2
        right_padding = h-w-left_padding
        image = np.pad(self.image,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        # load keypoints --> range [0, 1]
        fname = self._kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        pose, keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        pose = np.pad(pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        keypoints[:,0] += left_padding

        # load upper_cloth and lower body
        fname = self._parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        parsing = cv2.imread(f)[...,0:1]
        parsing = np.pad(parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))

        shoes_mask = (parsing==18).astype(np.uint8) + (parsing==19).astype(np.uint8)
        head_mask = (parsing==1).astype(np.uint8) + (parsing==2).astype(np.uint8) + \
                    (parsing==4).astype(np.uint8) + (parsing==13).astype(np.uint8)
        palm_mask = self.get_palm(keypoints, parsing)
        retain_mask = shoes_mask + palm_mask + head_mask

        neck_mask = (parsing==10).astype(np.uint8)
        face_mask = (parsing==13).astype(np.uint8)
        skin_mask = neck_mask + face_mask
        skin = skin_mask * image
        skin_r = skin[..., 0].reshape((-1))
        skin_g = skin[..., 1].reshape((-1))
        skin_b = skin[..., 2].reshape((-1))
        skin_r_valid_index = np.where(skin_r > 0)[0]
        skin_g_valid_index = np.where(skin_g > 0)[0]
        skin_b_valid_index = np.where(skin_b > 0)[0]
        skin_r_median = np.median(
            skin_r[skin_r_valid_index]) * np.ones_like(image[...,0:1])
        skin_g_median = np.median(
            skin_g[skin_g_valid_index]) * np.ones_like(image[...,0:1])
        skin_b_median = np.median(
            skin_b[skin_b_valid_index]) * np.ones_like(image[...,0:1])
        skin_average = np.concatenate([skin_r_median, skin_g_median, skin_b_median], axis=2)

        tops_mask = (parsing==5).astype(np.uint8) + (parsing==7).astype(np.uint8)
        dresses_mask = (parsing==6).astype(np.uint8)
        lower_pants_mask = (parsing==9).astype(np.uint8)
        lower_skirt_mask = (parsing==12).astype(np.uint8)

        if np.sum(lower_pants_mask) > np.sum(lower_skirt_mask):
            lower_pants_mask += lower_skirt_mask
            lower_skirt_mask *= 0
        else:
            lower_skirt_mask += lower_pants_mask
            lower_pants_mask *= 0
       
        if np.sum(dresses_mask) > 0:
            if np.sum(lower_pants_mask) > 0:
                tops_mask += dresses_mask
                dresses_mask *= 0
            else:
                if np.sum(dresses_mask) > (np.sum(tops_mask)+np.sum(lower_skirt_mask)):
                    dresses_mask += (tops_mask + lower_skirt_mask)
                    tops_mask *= 0
                    lower_skirt_mask *= 0
                else:
                    if np.sum(tops_mask) > np.sum(lower_skirt_mask):
                        lower_skirt_mask += dresses_mask
                    else:
                        tops_mask += dresses_mask
                    dresses_mask *= 0

        lower_clothes_mask = lower_skirt_mask + lower_pants_mask
        lower_clothes_image = lower_clothes_mask * image

        lower_bbox = self.mask_to_bbox(lower_clothes_mask.copy())
        lower_clothes_upper_bound = np.zeros_like(lower_clothes_mask[...,0:1])
        left_hip_kps = keypoints[11]
        right_hip_kps = keypoints[8]
        if left_hip_kps[2] > 0.05 and right_hip_kps[2] > 0.05:
            hip_width = np.linalg.norm(left_hip_kps[0:2] - right_hip_kps[0:2])
            middle_hip_y = (left_hip_kps[1]+right_hip_kps[1]) / 2
            # upper_bound_via_kps = int(middle_hip_y - (2 * hip_width / 3))
            upper_bound_via_kps = int(middle_hip_y - (3 * hip_width / 4))
            if lower_bbox is not None:
                upper_bound = lower_bbox[1]
                if upper_bound_via_kps < upper_bound:
                    upper_bound = upper_bound_via_kps
            else:
                upper_bound = upper_bound_via_kps
            lower_clothes_upper_bound[upper_bound:,...] += 255
        elif lower_bbox is not None:
            upper_bound = lower_bbox[1]
            lower_clothes_upper_bound[upper_bound:,...] += 255

        ##### clothes items
        fname = self._clothes_image_fnames[raw_idx]
        clothes_name = fname
        f = os.path.join(self._path, fname)
        self.clothes = np.array(PIL.Image.open(f))
        clothes = np.pad(self.clothes,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        fname = self._clothes_kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        clothes_pose, clothes_keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        clothes_pose = np.pad(clothes_pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        clothes_keypoints[:,0] += left_padding

        fname = self._clothes_parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        clothes_parsing = cv2.imread(f)[...,0:1]
        clothes_parsing = np.pad(clothes_parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))

        clothes_tops_mask = (clothes_parsing==5).astype(np.uint8) + (clothes_parsing==7).astype(np.uint8)
        clothes_dresses_mask = (clothes_parsing==6).astype(np.uint8)
        clothes_lower_pants_mask = (clothes_parsing==9).astype(np.uint8)
        clothes_lower_skirt_mask = (clothes_parsing==12).astype(np.uint8)

        if np.sum(clothes_lower_pants_mask) > np.sum(clothes_lower_skirt_mask):
            clothes_lower_pants_mask += clothes_lower_skirt_mask
            clothes_lower_skirt_mask *= 0
        else:
            clothes_lower_skirt_mask += clothes_lower_pants_mask
            clothes_lower_pants_mask *= 0
       
        if np.sum(clothes_dresses_mask) > 0:
            if np.sum(clothes_lower_pants_mask) > 0:
                clothes_tops_mask += clothes_dresses_mask
                clothes_dresses_mask *= 0
            else:
                if np.sum(clothes_dresses_mask) > (np.sum(clothes_tops_mask)+np.sum(clothes_lower_skirt_mask)):
                    clothes_dresses_mask += (clothes_tops_mask + clothes_lower_skirt_mask)
                    clothes_tops_mask *= 0
                    clothes_lower_skirt_mask *= 0
                else:
                    if np.sum(clothes_tops_mask) > np.sum(clothes_lower_skirt_mask):
                        clothes_lower_skirt_mask += clothes_dresses_mask
                    else:
                        clothes_tops_mask += clothes_dresses_mask
                    clothes_dresses_mask *= 0

        upper_clothes_mask = clothes_tops_mask + clothes_dresses_mask
        upper_clothes_image = upper_clothes_mask * clothes

        if np.sum(clothes_dresses_mask) > 0:
            lower_clothes_mask *= 0
            lower_pants_mask *= 0
            lower_skirt_mask *= 0
            lower_clothes_image *= 0
            lower_clothes_upper_bound *= 0

        upper_clothes_mask_rgb = np.concatenate([upper_clothes_mask,upper_clothes_mask,upper_clothes_mask],axis=2)
        lower_clothes_mask_rgb = np.concatenate([lower_clothes_mask,lower_clothes_mask,lower_clothes_mask],axis=2)
        upper_clothes_mask_rgb = upper_clothes_mask_rgb * 255
        lower_clothes_mask_rgb = lower_clothes_mask_rgb * 255
    
        sleeve_mask = None
        if self.use_sleeve_mask:
            fname = self._clothes_garment_parsing_fnames[raw_idx]
            f = os.path.join(self._path, fname)
            garment_parsing = cv2.imread(f)[...,0:1]
            garment_parsing = np.pad(garment_parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))
            sleeve_mask = (garment_parsing==10).astype(np.uint8) + (garment_parsing==11).astype(np.uint8)

        norm_img, norm_img_lower, denorm_upper_img, denorm_upper_img_wo_sleeve, denorm_lower_img = self.normalize(upper_clothes_image, \
            lower_clothes_image, upper_clothes_mask_rgb, lower_clothes_mask_rgb, sleeve_mask, clothes_keypoints, keypoints, 2)

        kernel = np.ones((8,8), dtype=np.uint8)
        denorm_lower_img_mask = cv2.erode(lower_clothes_mask_rgb, kernel, iterations=1)[...,0:1]
        denorm_lower_img_mask = (denorm_lower_img_mask==255).astype(np.uint8)
        denorm_lower_img = lower_clothes_image * denorm_lower_img_mask

        denorm_upper_img_mask = (np.sum(denorm_upper_img_wo_sleeve, axis=2, keepdims=True)>0).astype(np.uint8)
        upper_bbox = self.mask_to_bbox(denorm_upper_img_mask)
        if upper_bbox is not None:
            lower_bound = upper_bbox[3]
            lower_clothes_upper_bound[0:lower_bound,...] *= 0

        lower_label_map = np.ones_like(lower_clothes_mask)
        if np.sum(lower_pants_mask) > 0:
            lower_label_map *= 0
        elif np.sum(lower_skirt_mask) > 0:
            lower_label_map *= 1
        elif np.sum(clothes_dresses_mask) > 0:
            lower_label_map *= 2
        lower_label_map = lower_label_map / 2.0 * 255

        return image, clothes, pose, clothes_pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, \
                retain_mask, skin_average, lower_label_map, lower_clothes_upper_bound, person_name, clothes_name


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if not os.path.exists(os.path.join(self._path, fname)):
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    
    ############################ get palm mask start #########################################

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps  = [x1,y1,x2,y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1<cos2:
            kps.extend([x3,y3,x4,y4])
        else:
            kps.extend([x4,y4,x3,y3])

        kps = np.array(kps).reshape(1,-1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask
    
    def get_hand_mask(self, hand_keypoints):
        # shoulder, elbow, wrist    
        s_x,s_y,s_c = hand_keypoints[0]
        e_x,e_y,e_c = hand_keypoints[1]
        w_x,w_y,w_c = hand_keypoints[2]

        h, w = 512, 512
        up_mask = np.ones((512,512,1),dtype=np.float32)
        bottom_mask = np.ones((512,512,1),dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            kernel = np.ones((35,35),np.uint8)
            up_mask = cv2.dilate(up_mask,kernel,iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            kernel = np.ones((28,28),np.uint8)
            bottom_mask = cv2.dilate(bottom_mask,kernel,iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[...,np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)==2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, keypoints, parsing):
        left_hand_keypoints = keypoints[[5,6,7],:].copy()
        right_hand_keypoints = keypoints[[2,3,4],:].copy()

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############################ get palm mask end #########################################

    def draw_pose_from_cords(self, pose_joints, img_size, radius=5, draw_joints=True):
        colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
        if draw_joints:
            for i, p in enumerate(limbseq):
                f, t = p[0]-1, p[1]-1
                from_missing = pose_joints[f][2] < 0.05
                to_missing = pose_joints[t][2] < 0.05

                if from_missing or to_missing:
                    continue

                pf = pose_joints[f][0], pose_joints[f][1]
                pt = pose_joints[t][0], pose_joints[t][1]
                fx, fy = pf[1], pf[0]# max(pf[1], 0), max(pf[0], 0)
                tx, ty = pt[1], pt[0]# max(pt[1], 0), max(pt[0], 0)
                fx, fy = int(fx), int(fy)# int(min(fx, 255)), int(min(fy, 191))
                tx, ty = int(tx), int(ty)# int(min(tx, 255)), int(min(ty, 191))
                cv2.line(colors, (fy, fx), (ty, tx), kptcolors[i], 5)

        for i, joint in enumerate(pose_joints):
            if pose_joints[i][2] < 0.05:
                continue
            if i == 9 or i == 10 or i == 12 or i == 13:
                if (pose_joints[i][0] <= 0) or \
                   (pose_joints[i][1] <= 0) or \
                   (pose_joints[i][0] >= img_size[1]-50) or \
                   (pose_joints[i][1] >= img_size[0]-50):
                    pose_joints[i][2] = 0.01
                    continue
            pj = joint[0], joint[1]
            x, y = int(pj[1]), int(pj[0])# int(min(pj[1], 255)), int(min(pj[0], 191))
            xx, yy = circle(x, y, radius=radius, shape=img_size)
            colors[xx, yy] = kptcolors[i]
        
        return colors, pose_joints

    def get_joints(self, keypoints_path):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            keypoints = np.zeros((18,3))
        else:
            keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        color_joint, keypoints = self.draw_pose_from_cords(keypoints, (512, 320))
        return color_joint, keypoints

    def valid_joints(self, joint):
        return (joint >= 0.1).all()

    def get_crop(self, keypoints, bpart, order, wh, o_w, o_h, ar = 1.0):
        joints = keypoints
        bpart_indices = [order.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices][:, :2])
        # fall backs
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            if bpart[0] == "lhip" and bpart[1] == "lknee":
                bpart = ["lhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":
                bpart = ["rhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lknee" and bpart[1] == 'lankle':
                bpart = ["lknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rknee" and bpart[1] == 'rankle':
                bpart = ["rknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])

        if not self.valid_joints(joints[bpart_indices][:, 2]):
                return None, None

        if part_src.shape[0] == 1:
            # leg fallback
            # hip_bpart = ["lhip", "rhip"]
            # hip_indices = [order.index(bb) for bb in hip_bpart]

            # if not self.valid_joints(joints[hip_indices][:,2]):
            #     return None, None
            # a = part_src[0]
            # # b = np.float32([a[0],o_h - 1])
            # part_hip = np.float32(joints[hip_indices][:,:2])
            # leg_height = 2 * np.linalg.norm(part_hip[0]-part_hip[1])
            # b = np.float32([a[0],a[1]+leg_height])
            # part_src = np.float32([a,b])

            torso_bpart = ["lhip", "rhip", "cneck"]
            torso_indices = [order.index(bb) for bb in torso_bpart]

            if not self.valid_joints(joints[torso_indices][:,2]):
                return None, None
            
            a = part_src[0]
            if 'lhip' in bpart:
                invalid_label = 'lknee'
            elif 'rhip' in bpart:
                invalid_label = 'rknee'
            elif 'lknee' in bpart:
                invalid_label = 'lankle'
            elif 'rknee' in bpart:
                invalid_label = 'rankle'
            invalid_joint = joints[order.index(invalid_label)]

            part_torso = np.float32(joints[torso_indices][:,:2])
            torso_length = np.linalg.norm(part_torso[2]-part_torso[1]) + \
                        np.linalg.norm(part_torso[2]-part_torso[0])
            torso_length = torso_length / 2

            if invalid_joint[2] > 0:
                direction = (invalid_joint[0:2]-a) / np.linalg.norm(a-invalid_joint[0:2])
                # if 'hip' in bpart[0]:
                #     b = a + torso_length * direction * 0.85
                # elif 'knee' in bpart[0]:
                #     b = a + torso_length * direction * 0.8
                # if 'hip' in bpart[0]:
                #     b = a + torso_length * direction * 0.90
                # elif 'knee' in bpart[0]:
                #     b = a + torso_length * direction * 0.85
                if 'hip' in bpart[0]:
                    b = a + torso_length * direction * 0.85
                elif 'knee' in bpart[0]:
                    b = a + torso_length * direction * 0.80
            else:
                # b = np.float32([a[0],a[1]+torso_length])
                if 'hip' in bpart[0]:
                    b = np.float32([a[0],a[1]+torso_length * 0.85])
                elif 'knee' in bpart[0]:
                    b = np.float32([a[0],a[1]+torso_length * 0.80])

            part_src = np.float32([a,b])

        if part_src.shape[0] == 4:
            hip_seg = (part_src[2] - part_src[1]) / 4
            hip_l = part_src[1]
            hip_r = part_src[2]
            hip_l_new = hip_l - hip_seg
            hip_r_new = hip_r + hip_seg
            if hip_l_new[0] > 0 and hip_l_new[1] > 0 and hip_l_new[0] < o_w and hip_l_new[1] < o_h:
                part_src[1] = hip_l_new
            if hip_r_new[0] > 0 and hip_r_new[1] > 0 and hip_r_new[0] < o_w and hip_r_new[1] < o_h:
                part_src[2] = hip_r_new

            shoulder_seg = (part_src[3] - part_src[0]) / 5
            shoulder_l = part_src[0]
            shoulder_r = part_src[3]
            shoulder_l_new = shoulder_l - shoulder_seg
            shoulder_r_new = shoulder_r + shoulder_seg
            if shoulder_l_new[0] > 0 and shoulder_l_new[1] > 0 and shoulder_l_new[0] < o_w and shoulder_l_new[1] < o_h:
                part_src[0] = shoulder_l_new
            if shoulder_r_new[0] > 0 and shoulder_r_new[1] > 0 and shoulder_r_new[0] < o_w and shoulder_r_new[1] < o_h:
                part_src[3] = shoulder_r_new
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            shoulder_seg = (part_src[0] - part_src[1]) / 5
            shoulder_l = part_src[1]
            shoulder_r = part_src[0]
            shoulder_l_new = shoulder_l - shoulder_seg
            shoulder_r_new = shoulder_r + shoulder_seg
            if shoulder_l_new[0] > 0 and shoulder_l_new[1] > 0 and shoulder_l_new[0] < o_w and shoulder_l_new[1] < o_h:
                part_src[1] = shoulder_l_new
            if shoulder_r_new[0] > 0 and shoulder_r_new[1] > 0 and shoulder_r_new[0] < o_w and shoulder_r_new[1] < o_h:
                part_src[0] = shoulder_r_new  

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            if normal[1] > 0.0:
                normal = -normal

            a = part_src[0] + normal
            b = part_src[0]
            c = part_src[1]
            d = part_src[1] + normal

            part_height = (c[1]+b[1])/2 - (a[1]+d[1])/2
            a[1] += part_height/2
            d[1] += part_height/2
            part_src = np.float32([d,c,b,a])
        else:
            assert part_src.shape[0] == 2                           
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            if 'rhip' in bpart or 'rknee' in bpart:
                # a = a + alpha*normal*1.5
                # d = d + alpha*normal*1.5
                a = a + alpha*normal*1.0
                d = d + alpha*normal*1.0
            if 'lhip' in bpart or 'lknee' in bpart:
                b = b - alpha*normal*1.0
                c = c - alpha*normal*1.0
            if 'relbow' in bpart or 'rwrist' in bpart:
                a = a + alpha*normal*0.45
                d = d + alpha*normal*0.45
                b = b - alpha*normal*0.1
                c = c - alpha*normal*0.1
            if 'lelbow' in bpart or 'lwrist' in bpart:
                a = a + alpha*normal*0.1
                d = d + alpha*normal*0.1
                b = b - alpha*normal*0.45
                c = c - alpha*normal*0.45
            part_src = np.float32([a,d,c,b])

        dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        M_inv = cv2.getPerspectiveTransform(part_dst,part_src)
        return M, M_inv

    def mask_to_bbox(self, mask):
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        site = np.where(mask > 0)
        if len(site[0]) > 0 and len(site[1]) > 0:
            bbox = [np.min(site[1]), np.min(site[0]),
                    np.max(site[1]), np.max(site[0])]
            return bbox
        else:
            return None

    def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask, sleeve_mask, clothes_keypoints, \
                    person_keypoints, box_factor):
        h, w = upper_img.shape[:2]
        o_h, o_w = h, w
        h = h // 2**box_factor
        w = w // 2**box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
                ["rshoulder","rhip","lhip","lshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder","lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder","relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["lknee", "lankle"],
                ["rhip", "rknee"],
                ["rknee", "rankle"]]

        order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 
                'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee',  
                'lankle', 'reye', 'leye', 'rear', 'lear']
        # ar = 0.5

        part_imgs = list()
        part_imgs_lower = list()
        part_clothes_masks = list()
        part_clothes_masks_lower = list()

        denorm_upper_img = np.zeros_like(upper_img)
        denorm_upper_img_wo_sleeve = np.zeros_like(upper_img)
        denorm_lower_img = np.zeros_like(upper_img)
        kernel = np.ones((8,8),np.uint8)

        for ii, bpart in enumerate(bparts):
            if ii < 6:
                ar = 0.5
            else:
                ar = 0.4

            part_img = np.zeros((h, w, 3)).astype(np.uint8)
            part_img_lower = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask_lower = np.zeros((h,w,3)).astype(np.uint8)
            
            clothes_M, clothes_M_inv = self.get_crop(clothes_keypoints, bpart, order, wh, o_w, o_h, ar)
            person_M, person_M_inv = self.get_crop(person_keypoints, bpart, order, wh, o_w, o_h, ar)

            if clothes_M is not None:
                if ii == 2 or ii == 3 or ii == 4 or ii == 5:
                    if sleeve_mask is not None:
                        part_img = cv2.warpPerspective(upper_img*sleeve_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask*sleeve_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    else:
                        part_img = cv2.warpPerspective(upper_img, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                else:
                    if sleeve_mask is not None:
                        part_img = cv2.warpPerspective(upper_img*(1-sleeve_mask), clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask*(1-sleeve_mask), clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    else:
                        part_img = cv2.warpPerspective(upper_img, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)

                if person_M_inv is not None:
                    denorm_patch = cv2.warpPerspective(part_img, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                    # part_img = cv2.warpPerspective(denorm_patch, person_M, (w,h), borderMode=cv2.BORDER_CONSTANT)

                    denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                    denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)[...,np.newaxis]
                    denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)

                    denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)
                    if ii != 2 and ii != 3 and ii != 4 and ii != 5:
                        denorm_upper_img_wo_sleeve = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img_wo_sleeve * (1-denorm_clothes_mask_patch)

            if ii == 0 or ii >= 6:
                if person_M is not None:
                    part_img_lower = cv2.warpPerspective(lower_img, person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    part_clothes_mask_lower = cv2.warpPerspective(lower_clothes_mask, person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)

                    if person_M_inv is not None:
                        denorm_patch_lower = cv2.warpPerspective(part_img_lower, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)                        
                        # part_img_lower = cv2.warpPerspective(denorm_patch_lower, person_M, (w,h), borderMode=cv2.BORDER_CONSTANT)
                        
                        denorm_clothes_mask_patch_lower = cv2.warpPerspective(part_clothes_mask_lower, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                        denorm_clothes_mask_patch_lower = cv2.erode(denorm_clothes_mask_patch_lower, kernel, iterations=1)[...,np.newaxis]
                        denorm_clothes_mask_patch_lower = (denorm_clothes_mask_patch_lower==255).astype(np.uint8)

                        denorm_lower_img = denorm_patch_lower * denorm_clothes_mask_patch_lower + denorm_lower_img * (1-denorm_clothes_mask_patch_lower)

            part_imgs.append(part_img)
            part_clothes_masks.append(part_clothes_mask)
            if ii == 0 or ii >= 6:
                part_imgs_lower.append(part_img_lower)
                part_clothes_masks_lower.append(part_clothes_mask_lower)

        upper_torso_mask = (np.sum(part_clothes_masks[0],axis=2,keepdims=True)>0).astype(np.uint8)
        upper_left_hip_mask = (np.sum(part_clothes_masks[6],axis=2,keepdims=True)>0).astype(np.uint8)
        upper_right_hip_mask = (np.sum(part_clothes_masks[8],axis=2,keepdims=True)>0).astype(np.uint8)
        
        part_imgs_lower[0] = part_imgs_lower[0] * (1-upper_torso_mask)
        part_imgs_lower[1] = part_imgs_lower[1] * (1-upper_left_hip_mask)
        part_imgs_lower[3] = part_imgs_lower[3] * (1-upper_right_hip_mask)
        part_clothes_masks_lower[0] = part_clothes_masks_lower[0] * (1-upper_torso_mask)
        part_clothes_masks_lower[1] = part_clothes_masks_lower[1] * (1-upper_left_hip_mask)
        part_clothes_masks_lower[3] = part_clothes_masks_lower[3] * (1-upper_right_hip_mask)
        
        left_top_sleeve_mask = part_clothes_masks[2]
        right_top_sleeve_mask = part_clothes_masks[4]
        left_bottom_sleeve_mask = part_clothes_masks[3]
        right_bottom_sleeve_mask = part_clothes_masks[5]

        if np.sum(left_top_sleeve_mask) == 0 and np.sum(right_top_sleeve_mask) > 0:
            right_top_sleeve = part_imgs[4]
            left_top_sleeve = cv2.flip(right_top_sleeve,1)
            left_top_sleeve_mask = cv2.flip(right_top_sleeve_mask,1)
            part_imgs[2] = left_top_sleeve
            part_clothes_masks[2] = left_top_sleeve_mask
        elif np.sum(right_top_sleeve_mask) == 0 and np.sum(left_top_sleeve_mask) > 0:
            left_top_sleeve = part_imgs[2]
            right_top_sleeve = cv2.flip(left_top_sleeve,1)
            right_top_sleeve_mask = cv2.flip(left_top_sleeve_mask,1)
            part_imgs[4] = right_top_sleeve
            part_clothes_masks[4] = right_top_sleeve_mask


        if np.sum(left_bottom_sleeve_mask) == 0 and np.sum(right_bottom_sleeve_mask) > 0:
            right_bottom_sleeve = part_imgs[3]
            left_bottom_sleeve = cv2.flip(right_bottom_sleeve, 1)
            left_bottom_sleeve_mask = cv2.flip(right_bottom_sleeve_mask, 1)
            part_imgs[3] = left_bottom_sleeve
            part_clothes_masks[3] = left_bottom_sleeve_mask
        elif np.sum(right_bottom_sleeve_mask) == 0 and np.sum(left_bottom_sleeve_mask) > 0:
            left_bottom_sleeve = part_imgs[5]
            right_bottom_sleeve = cv2.flip(left_bottom_sleeve, 1)
            right_bottom_sleeve_mask = cv2.flip(left_bottom_sleeve_mask, 1)
            part_imgs[5] = right_bottom_sleeve
            part_clothes_masks[5] = right_bottom_sleeve_mask

        img = np.concatenate(part_imgs, axis = 2)
        img_lower = np.concatenate(part_imgs_lower, axis=2)

        return img, img_lower, denorm_upper_img, denorm_upper_img_wo_sleeve, denorm_lower_img

    def __getitem__(self, idx):
        image, clothes, pose, clothes_pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, \
            retain_mask, skin_average, lower_label_map, lower_clothes_upper_bound, \
            person_name, clothes_name = self._load_raw_image(self._raw_idx[idx])

        image = image.transpose(2, 0, 1)                    # HWC => CHW
        clothes = clothes.transpose(2,0,1)
        pose = pose.transpose(2, 0, 1)                      # HWC => CHW
        clothes_pose = clothes_pose.transpose(2,0,1)
        norm_img = norm_img.transpose(2, 0, 1)
        norm_img_lower = norm_img_lower.transpose(2,0,1)
        denorm_upper_img = denorm_upper_img.transpose(2,0,1)
        denorm_lower_img = denorm_lower_img.transpose(2,0,1)
        denorm_upper_mask = (np.sum(denorm_upper_img, axis=0, keepdims=True)>0).astype(np.uint8)
        denorm_lower_mask = (np.sum(denorm_lower_img, axis=0, keepdims=True)>0).astype(np.uint8)

        skin_average = skin_average.transpose(2,0,1)
        retain_mask = retain_mask.transpose(2,0,1)
        lower_label_map = lower_label_map.transpose(2,0,1)
        lower_clothes_upper_bound = lower_clothes_upper_bound.transpose(2,0,1)

        return image.copy(), clothes.copy(), pose.copy(), clothes_pose.copy(), norm_img.copy(), norm_img_lower.copy(), \
               denorm_upper_img.copy(), denorm_lower_img.copy(), denorm_upper_mask.copy(), denorm_lower_mask.copy(), \
               retain_mask.copy(), skin_average.copy(), lower_label_map.copy(), lower_clothes_upper_bound.copy(),\
               person_name, clothes_name


class UvitonDatasetFull_512_test_lower(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        test_txt,
        use_sleeve_mask,
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.use_sleeve_mask = use_sleeve_mask

        if os.path.isdir(self._path):
            self._type = 'dir' 
            self._image_fnames = []
            self._kpt_fnames = []
            self._parsing_fnames = []

            self._clothes_image_fnames = []
            self._clothes_kpt_fnames = []
            self._clothes_parsing_fnames = []
            self._garment_parsing_fnames = []

            txt_path = os.path.join(self._path, test_txt)
            with open(txt_path, 'r') as f:
                for ii, line in enumerate(f.readlines()):
                    clothes_name, person_name = line.strip().split()

                    self._image_fnames.append(os.path.join('image', person_name))
                    self._kpt_fnames.append(os.path.join('keypoints', person_name[:-4]+'_keypoints.json'))
                    self._garment_parsing_fnames.append(os.path.join('garment_parsing', person_name[:-4]+'.png'))
                    self._parsing_fnames.append(os.path.join('parsing',person_name.replace('.jpg','.png')))

                    self._clothes_image_fnames.append(os.path.join('image', clothes_name))
                    self._clothes_kpt_fnames.append(os.path.join('keypoints', clothes_name[:-4]+'_keypoints.json'))
                    self._clothes_parsing_fnames.append(os.path.join('parsing',clothes_name.replace('.jpg','.png')))
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        im_shape = list((self._load_raw_image(0))[0].shape)
        raw_shape = [len(self._image_fnames)] + [im_shape[2], im_shape[0], im_shape[1]]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        # load images --> range [0, 255]
        fname = self._image_fnames[raw_idx]
        person_name = fname
        f = os.path.join(self._path, fname)
        self.image = np.array(PIL.Image.open(f))
        im_shape = self.image.shape
        h, w = im_shape[0], im_shape[1]
        left_padding = (h-w) // 2
        right_padding = h-w-left_padding
        image = np.pad(self.image,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        # load keypoints --> range [0, 1]
        fname = self._kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        pose, keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        pose = np.pad(pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        keypoints[:,0] += left_padding

        # load upper_cloth and lower body
        fname = self._parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        parsing = cv2.imread(f)[...,0:1]
        parsing = np.pad(parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))

        shoes_mask = (parsing==18).astype(np.uint8) + (parsing==19).astype(np.uint8)
        head_mask = (parsing==1).astype(np.uint8) + (parsing==2).astype(np.uint8) + \
                    (parsing==4).astype(np.uint8) + (parsing==13).astype(np.uint8)
        palm_mask = self.get_palm(keypoints, parsing)
        retain_mask = shoes_mask + palm_mask + head_mask

        neck_mask = (parsing==10).astype(np.uint8)
        face_mask = (parsing==13).astype(np.uint8)
        skin_mask = neck_mask + face_mask
        skin = skin_mask * image
        skin_r = skin[..., 0].reshape((-1))
        skin_g = skin[..., 1].reshape((-1))
        skin_b = skin[..., 2].reshape((-1))
        skin_r_valid_index = np.where(skin_r > 0)[0]
        skin_g_valid_index = np.where(skin_g > 0)[0]
        skin_b_valid_index = np.where(skin_b > 0)[0]
        skin_r_median = np.median(
            skin_r[skin_r_valid_index]) * np.ones_like(image[...,0:1])
        skin_g_median = np.median(
            skin_g[skin_g_valid_index]) * np.ones_like(image[...,0:1])
        skin_b_median = np.median(
            skin_b[skin_b_valid_index]) * np.ones_like(image[...,0:1])
        skin_average = np.concatenate([skin_r_median, skin_g_median, skin_b_median], axis=2)

        tops_mask = (parsing==5).astype(np.uint8) + (parsing==7).astype(np.uint8)
        dresses_mask = (parsing==6).astype(np.uint8)
        lower_pants_mask = (parsing==9).astype(np.uint8)
        lower_skirt_mask = (parsing==12).astype(np.uint8)

        if np.sum(lower_pants_mask) > np.sum(lower_skirt_mask):
            lower_pants_mask += lower_skirt_mask
            lower_skirt_mask *= 0
        else:
            lower_skirt_mask += lower_pants_mask
            lower_pants_mask *= 0
       
        if np.sum(dresses_mask) > 0:
            if np.sum(lower_pants_mask) > 0:
                tops_mask += dresses_mask
                dresses_mask *= 0
            else:
                if np.sum(dresses_mask) > (np.sum(tops_mask)+np.sum(lower_skirt_mask)):
                    dresses_mask += (tops_mask + lower_skirt_mask)
                    tops_mask *= 0
                    lower_skirt_mask *= 0
                else:
                    if np.sum(tops_mask) > np.sum(lower_skirt_mask):
                        lower_skirt_mask += dresses_mask
                    else:
                        tops_mask += dresses_mask
                    dresses_mask *= 0

        upper_clothes_mask = tops_mask + dresses_mask
        upper_clothes_image = upper_clothes_mask * image

        lower_clothes_mask = lower_skirt_mask + lower_pants_mask
        lower_bbox = self.mask_to_bbox(lower_clothes_mask.copy())
        lower_clothes_upper_bound = np.zeros_like(lower_clothes_mask[...,0:1])
        if lower_bbox is not None:
            upper_bound = lower_bbox[1]
            lower_clothes_upper_bound[upper_bound:,...] += 255

        sleeve_mask = None
        if self.use_sleeve_mask:
            fname = self._garment_parsing_fnames[raw_idx]
            f = os.path.join(self._path, fname)
            garment_parsing = cv2.imread(f)[...,0:1]
            garment_parsing = np.pad(garment_parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))
            sleeve_mask = (garment_parsing==10).astype(np.uint8) + (garment_parsing==11).astype(np.uint8)

        ##### clothes items
        fname = self._clothes_image_fnames[raw_idx]
        clothes_name = fname
        f = os.path.join(self._path, fname)
        self.clothes = np.array(PIL.Image.open(f))
        clothes = np.pad(self.clothes,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        fname = self._clothes_kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        clothes_pose, clothes_keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        clothes_pose = np.pad(clothes_pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        clothes_keypoints[:,0] += left_padding

        fname = self._clothes_parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        clothes_parsing = cv2.imread(f)[...,0:1]
        clothes_parsing = np.pad(clothes_parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))

        clothes_tops_mask = (clothes_parsing==5).astype(np.uint8) + (clothes_parsing==7).astype(np.uint8)
        clothes_dresses_mask = (clothes_parsing==6).astype(np.uint8)
        clothes_lower_pants_mask = (clothes_parsing==9).astype(np.uint8)
        clothes_lower_skirt_mask = (clothes_parsing==12).astype(np.uint8)

        if np.sum(clothes_lower_pants_mask) > np.sum(clothes_lower_skirt_mask):
            clothes_lower_pants_mask += clothes_lower_skirt_mask
            clothes_lower_skirt_mask *= 0
        else:
            clothes_lower_skirt_mask += clothes_lower_pants_mask
            clothes_lower_pants_mask *= 0
       
        if np.sum(clothes_dresses_mask) > 0:
            if np.sum(clothes_lower_pants_mask) > 0:
                clothes_tops_mask += clothes_dresses_mask
                clothes_dresses_mask *= 0
            else:
                if np.sum(clothes_dresses_mask) > (np.sum(clothes_tops_mask)+np.sum(clothes_lower_skirt_mask)):
                    clothes_dresses_mask += (clothes_tops_mask + clothes_lower_skirt_mask)
                    clothes_tops_mask *= 0
                    clothes_lower_skirt_mask *= 0
                else:
                    if np.sum(clothes_tops_mask) > np.sum(clothes_lower_skirt_mask):
                        clothes_lower_skirt_mask += clothes_dresses_mask
                    else:
                        clothes_tops_mask += clothes_dresses_mask
                    clothes_dresses_mask *= 0

        lower_clothes_mask = clothes_lower_skirt_mask + clothes_lower_pants_mask
        lower_clothes_image = lower_clothes_mask * clothes

        if np.sum(dresses_mask) > 0:
            clothes_lower_skirt_mask *= 0
            clothes_lower_pants_mask *= 0
            lower_clothes_mask *= 0
            lower_clothes_image *= 0
            lower_clothes_upper_bound *= 0

        upper_clothes_mask_rgb = np.concatenate([upper_clothes_mask,upper_clothes_mask,upper_clothes_mask],axis=2)
        lower_clothes_mask_rgb = np.concatenate([lower_clothes_mask,lower_clothes_mask,lower_clothes_mask],axis=2)
        upper_clothes_mask_rgb = upper_clothes_mask_rgb * 255
        lower_clothes_mask_rgb = lower_clothes_mask_rgb * 255
    
        norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img = self.normalize(upper_clothes_image, \
            lower_clothes_image, upper_clothes_mask_rgb, lower_clothes_mask_rgb, sleeve_mask, \
            clothes_keypoints, keypoints, 2)

        kernel = np.ones((8,8), dtype=np.uint8)
        denorm_upper_img_mask = cv2.erode(upper_clothes_mask_rgb, kernel, iterations=1)[...,0:1]
        denorm_upper_img_mask = (denorm_upper_img_mask==255).astype(np.uint8)
        denorm_upper_img = upper_clothes_image * denorm_upper_img_mask

        lower_label_map = np.ones_like(lower_clothes_mask)
        if np.sum(clothes_lower_pants_mask) > 0:
            lower_label_map *= 0
        elif np.sum(clothes_lower_skirt_mask) > 0:
            lower_label_map *= 1
        elif np.sum(dresses_mask) > 0:
            lower_label_map *= 2
        lower_label_map = lower_label_map / 2.0 * 255

        return image, clothes, pose, clothes_pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, \
                retain_mask, skin_average, lower_label_map, lower_clothes_upper_bound, person_name, clothes_name


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if not os.path.exists(os.path.join(self._path, fname)):
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    
    ############################ get palm mask start #########################################

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps  = [x1,y1,x2,y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1<cos2:
            kps.extend([x3,y3,x4,y4])
        else:
            kps.extend([x4,y4,x3,y3])

        kps = np.array(kps).reshape(1,-1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask
    
    def get_hand_mask(self, hand_keypoints):
        # shoulder, elbow, wrist    
        s_x,s_y,s_c = hand_keypoints[0]
        e_x,e_y,e_c = hand_keypoints[1]
        w_x,w_y,w_c = hand_keypoints[2]

        h, w = 512, 512
        up_mask = np.ones((512,512,1),dtype=np.float32)
        bottom_mask = np.ones((512,512,1),dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            kernel = np.ones((35,35),np.uint8)
            up_mask = cv2.dilate(up_mask,kernel,iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            kernel = np.ones((28,28),np.uint8)
            bottom_mask = cv2.dilate(bottom_mask,kernel,iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[...,np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)==2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, keypoints, parsing):
        left_hand_keypoints = keypoints[[5,6,7],:].copy()
        right_hand_keypoints = keypoints[[2,3,4],:].copy()

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############################ get palm mask end #########################################

    def draw_pose_from_cords(self, pose_joints, img_size, radius=5, draw_joints=True):
        colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
        if draw_joints:
            for i, p in enumerate(limbseq):
                f, t = p[0]-1, p[1]-1
                from_missing = pose_joints[f][2] < 0.05
                to_missing = pose_joints[t][2] < 0.05

                if from_missing or to_missing:
                    continue

                pf = pose_joints[f][0], pose_joints[f][1]
                pt = pose_joints[t][0], pose_joints[t][1]
                fx, fy = pf[1], pf[0]# max(pf[1], 0), max(pf[0], 0)
                tx, ty = pt[1], pt[0]# max(pt[1], 0), max(pt[0], 0)
                fx, fy = int(fx), int(fy)# int(min(fx, 255)), int(min(fy, 191))
                tx, ty = int(tx), int(ty)# int(min(tx, 255)), int(min(ty, 191))
                cv2.line(colors, (fy, fx), (ty, tx), kptcolors[i], 5)

        for i, joint in enumerate(pose_joints):
            if pose_joints[i][2] < 0.05:
                continue
            if i == 9 or i == 10 or i == 12 or i == 13:
                if (pose_joints[i][0] <= 0) or \
                   (pose_joints[i][1] <= 0) or \
                   (pose_joints[i][0] >= img_size[1]-50) or \
                   (pose_joints[i][1] >= img_size[0]-50):
                    pose_joints[i][2] = 0.01
                    continue
            pj = joint[0], joint[1]
            x, y = int(pj[1]), int(pj[0])# int(min(pj[1], 255)), int(min(pj[0], 191))
            xx, yy = circle(x, y, radius=radius, shape=img_size)
            colors[xx, yy] = kptcolors[i]
        
        return colors, pose_joints

    def get_joints(self, keypoints_path):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            keypoints = np.zeros((18,3))
        else:
            keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        color_joint, keypoints = self.draw_pose_from_cords(keypoints, (512, 320))
        return color_joint, keypoints

    def valid_joints(self, joint):
        return (joint >= 0.1).all()

    def get_crop(self, keypoints, bpart, order, wh, o_w, o_h, ar = 1.0):
        joints = keypoints
        bpart_indices = [order.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices][:, :2])
        # fall backs
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            if bpart[0] == "lhip" and bpart[1] == "lknee":
                bpart = ["lhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":
                bpart = ["rhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lknee" and bpart[1] == 'lankle':
                bpart = ["lknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rknee" and bpart[1] == 'rankle':
                bpart = ["rknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])

        if not self.valid_joints(joints[bpart_indices][:, 2]):
                return None, None

        if part_src.shape[0] == 1:
            # leg fallback
            # hip_bpart = ["lhip", "rhip"]
            # hip_indices = [order.index(bb) for bb in hip_bpart]

            # if not self.valid_joints(joints[hip_indices][:,2]):
            #     return None, None
            # a = part_src[0]
            # # b = np.float32([a[0],o_h - 1])
            # part_hip = np.float32(joints[hip_indices][:,:2])
            # leg_height = 2 * np.linalg.norm(part_hip[0]-part_hip[1])
            # b = np.float32([a[0],a[1]+leg_height])
            # part_src = np.float32([a,b])

            torso_bpart = ["lhip", "rhip", "cneck"]
            torso_indices = [order.index(bb) for bb in torso_bpart]

            if not self.valid_joints(joints[torso_indices][:,2]):
                return None, None
            
            a = part_src[0]
            if 'lhip' in bpart:
                invalid_label = 'lknee'
            elif 'rhip' in bpart:
                invalid_label = 'rknee'
            elif 'lknee' in bpart:
                invalid_label = 'lankle'
            elif 'rknee' in bpart:
                invalid_label = 'rankle'
            invalid_joint = joints[order.index(invalid_label)]

            part_torso = np.float32(joints[torso_indices][:,:2])
            torso_length = np.linalg.norm(part_torso[2]-part_torso[1]) + \
                        np.linalg.norm(part_torso[2]-part_torso[0])
            torso_length = torso_length / 2

            if invalid_joint[2] > 0:
                direction = (invalid_joint[0:2]-a) / np.linalg.norm(a-invalid_joint[0:2])
                # if 'hip' in bpart[0]:
                #     b = a + torso_length * direction * 0.85
                # elif 'knee' in bpart[0]:
                #     b = a + torso_length * direction * 0.8
                # if 'hip' in bpart[0]:
                #     b = a + torso_length * direction * 0.90
                # elif 'knee' in bpart[0]:
                #     b = a + torso_length * direction * 0.85
                if 'hip' in bpart[0]:
                    b = a + torso_length * direction * 0.85
                elif 'knee' in bpart[0]:
                    b = a + torso_length * direction * 0.80
            else:
                # b = np.float32([a[0],a[1]+torso_length])
                if 'hip' in bpart[0]:
                    b = np.float32([a[0],a[1]+torso_length * 0.85])
                elif 'knee' in bpart[0]:
                    b = np.float32([a[0],a[1]+torso_length * 0.80])

            part_src = np.float32([a,b])

        if part_src.shape[0] == 4:
            hip_seg = (part_src[2] - part_src[1]) / 4
            hip_l = part_src[1]
            hip_r = part_src[2]
            hip_l_new = hip_l - hip_seg
            hip_r_new = hip_r + hip_seg
            if hip_l_new[0] > 0 and hip_l_new[1] > 0 and hip_l_new[0] < o_w and hip_l_new[1] < o_h:
                part_src[1] = hip_l_new
            if hip_r_new[0] > 0 and hip_r_new[1] > 0 and hip_r_new[0] < o_w and hip_r_new[1] < o_h:
                part_src[2] = hip_r_new

            shoulder_seg = (part_src[3] - part_src[0]) / 5
            shoulder_l = part_src[0]
            shoulder_r = part_src[3]
            shoulder_l_new = shoulder_l - shoulder_seg
            shoulder_r_new = shoulder_r + shoulder_seg
            if shoulder_l_new[0] > 0 and shoulder_l_new[1] > 0 and shoulder_l_new[0] < o_w and shoulder_l_new[1] < o_h:
                part_src[0] = shoulder_l_new
            if shoulder_r_new[0] > 0 and shoulder_r_new[1] > 0 and shoulder_r_new[0] < o_w and shoulder_r_new[1] < o_h:
                part_src[3] = shoulder_r_new
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            shoulder_seg = (part_src[0] - part_src[1]) / 5
            shoulder_l = part_src[1]
            shoulder_r = part_src[0]
            shoulder_l_new = shoulder_l - shoulder_seg
            shoulder_r_new = shoulder_r + shoulder_seg
            if shoulder_l_new[0] > 0 and shoulder_l_new[1] > 0 and shoulder_l_new[0] < o_w and shoulder_l_new[1] < o_h:
                part_src[1] = shoulder_l_new
            if shoulder_r_new[0] > 0 and shoulder_r_new[1] > 0 and shoulder_r_new[0] < o_w and shoulder_r_new[1] < o_h:
                part_src[0] = shoulder_r_new  

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            if normal[1] > 0.0:
                normal = -normal

            a = part_src[0] + normal
            b = part_src[0]
            c = part_src[1]
            d = part_src[1] + normal

            part_height = (c[1]+b[1])/2 - (a[1]+d[1])/2
            a[1] += part_height/2
            d[1] += part_height/2
            part_src = np.float32([d,c,b,a])
        else:
            assert part_src.shape[0] == 2                           
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            if 'rhip' in bpart or 'rknee' in bpart:
                # a = a + alpha*normal*1.5
                # d = d + alpha*normal*1.5
                a = a + alpha*normal*1.0
                d = d + alpha*normal*1.0
            if 'lhip' in bpart or 'lknee' in bpart:
                b = b - alpha*normal*1.0
                c = c - alpha*normal*1.0
            if 'relbow' in bpart or 'rwrist' in bpart:
                a = a + alpha*normal*0.45
                d = d + alpha*normal*0.45
                b = b - alpha*normal*0.1
                c = c - alpha*normal*0.1
            if 'lelbow' in bpart or 'lwrist' in bpart:
                a = a + alpha*normal*0.1
                d = d + alpha*normal*0.1
                b = b - alpha*normal*0.45
                c = c - alpha*normal*0.45
            part_src = np.float32([a,d,c,b])

        dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        M_inv = cv2.getPerspectiveTransform(part_dst,part_src)
        return M, M_inv

    def mask_to_bbox(self, mask):
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        site = np.where(mask > 0)
        if len(site[0]) > 0 and len(site[1]) > 0:
            bbox = [np.min(site[1]), np.min(site[0]),
                    np.max(site[1]), np.max(site[0])]
            return bbox
        else:
            return None

    def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask, sleeve_mask, \
                    clothes_keypoints, person_keypoints, box_factor):
        h, w = upper_img.shape[:2]
        o_h, o_w = h, w
        h = h // 2**box_factor
        w = w // 2**box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
                ["rshoulder","rhip","lhip","lshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder","lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder","relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["lknee", "lankle"],
                ["rhip", "rknee"],
                ["rknee", "rankle"]]

        order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 
                'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee',  
                'lankle', 'reye', 'leye', 'rear', 'lear']

        part_imgs = list()
        part_imgs_lower = list()
        part_clothes_masks = list()
        part_clothes_masks_lower = list()

        denorm_upper_img = np.zeros_like(upper_img)
        denorm_lower_img = np.zeros_like(upper_img)
        kernel = np.ones((5,5),np.uint8)

        for ii, bpart in enumerate(bparts):
            if ii < 6:
                ar = 0.5
            else:
                ar = 0.4

            part_img = np.zeros((h, w, 3)).astype(np.uint8)
            part_img_lower = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask_lower = np.zeros((h,w,3)).astype(np.uint8)
            
            clothes_M, clothes_M_inv = self.get_crop(clothes_keypoints, bpart, order, wh, o_w, o_h, ar)
            person_M, person_M_inv = self.get_crop(person_keypoints, bpart, order, wh, o_w, o_h, ar)

            if person_M is not None:
                if ii == 2 or ii == 3 or ii == 4 or ii == 5:
                    if sleeve_mask is not None:
                        part_img = cv2.warpPerspective(upper_img*sleeve_mask, person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask*sleeve_mask, person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    else:
                        part_img = cv2.warpPerspective(upper_img, person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                else:
                    if sleeve_mask is not None:
                        part_img = cv2.warpPerspective(upper_img*(1-sleeve_mask), person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask*(1-sleeve_mask), person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    else:
                        part_img = cv2.warpPerspective(upper_img, person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, person_M, (w,h), borderMode = cv2.BORDER_CONSTANT)

                if person_M_inv is not None:
                    denorm_patch = cv2.warpPerspective(part_img, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)                    
                    # part_img = cv2.warpPerspective(denorm_patch, person_M, (w,h), borderMode=cv2.BORDER_CONSTANT)

                    denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                    denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)[...,np.newaxis]
                    denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)

                    denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)

            if ii == 0 or ii >= 6:
                if clothes_M is not None:
                    part_img_lower = cv2.warpPerspective(lower_img, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)
                    part_clothes_mask_lower = cv2.warpPerspective(lower_clothes_mask, clothes_M, (w,h), borderMode = cv2.BORDER_CONSTANT)

                    if person_M_inv is not None:
                        denorm_patch_lower = cv2.warpPerspective(part_img_lower, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)                        
                        # part_img_lower = cv2.warpPerspective(denorm_patch_lower, person_M, (w,h), borderMode=cv2.BORDER_CONSTANT)
                        
                        denorm_clothes_mask_patch_lower = cv2.warpPerspective(part_clothes_mask_lower, person_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                        denorm_clothes_mask_patch_lower = cv2.erode(denorm_clothes_mask_patch_lower, kernel, iterations=1)[...,np.newaxis]
                        denorm_clothes_mask_patch_lower = (denorm_clothes_mask_patch_lower==255).astype(np.uint8)

                        denorm_lower_img = denorm_patch_lower * denorm_clothes_mask_patch_lower + denorm_lower_img * (1-denorm_clothes_mask_patch_lower)

            part_imgs.append(part_img)
            part_clothes_masks.append(part_clothes_mask)
            if ii == 0 or ii >= 6:
                part_imgs_lower.append(part_img_lower)
                part_clothes_masks_lower.append(part_clothes_mask_lower)

        upper_torso_mask = (np.sum(part_clothes_masks[0],axis=2,keepdims=True)>0).astype(np.uint8)
        upper_left_hip_mask = (np.sum(part_clothes_masks[6],axis=2,keepdims=True)>0).astype(np.uint8)
        upper_right_hip_mask = (np.sum(part_clothes_masks[8],axis=2,keepdims=True)>0).astype(np.uint8)
        
        part_imgs_lower[0] = part_imgs_lower[0] * (1-upper_torso_mask)
        part_imgs_lower[1] = part_imgs_lower[1] * (1-upper_left_hip_mask)
        part_imgs_lower[3] = part_imgs_lower[3] * (1-upper_right_hip_mask)
        part_clothes_masks_lower[0] = part_clothes_masks_lower[0] * (1-upper_torso_mask)
        part_clothes_masks_lower[1] = part_clothes_masks_lower[1] * (1-upper_left_hip_mask)
        part_clothes_masks_lower[3] = part_clothes_masks_lower[3] * (1-upper_right_hip_mask)

        left_top_sleeve_mask = part_clothes_masks[2]
        right_top_sleeve_mask = part_clothes_masks[4]
        left_bottom_sleeve_mask = part_clothes_masks[3]
        right_bottom_sleeve_mask = part_clothes_masks[5]

        if np.sum(left_top_sleeve_mask) == 0 and np.sum(right_top_sleeve_mask) > 0:
            right_top_sleeve = part_imgs[4]
            left_top_sleeve = cv2.flip(right_top_sleeve,1)
            left_top_sleeve_mask = cv2.flip(right_top_sleeve_mask,1)
            part_imgs[2] = left_top_sleeve
            part_clothes_masks[2] = left_top_sleeve_mask
        elif np.sum(right_top_sleeve_mask) == 0 and np.sum(left_top_sleeve_mask) > 0:
            left_top_sleeve = part_imgs[2]
            right_top_sleeve = cv2.flip(left_top_sleeve,1)
            right_top_sleeve_mask = cv2.flip(left_top_sleeve_mask,1)
            part_imgs[4] = right_top_sleeve
            part_clothes_masks[4] = right_top_sleeve_mask


        if np.sum(left_bottom_sleeve_mask) == 0 and np.sum(right_bottom_sleeve_mask) > 0:
            right_bottom_sleeve = part_imgs[3]
            left_bottom_sleeve = cv2.flip(right_bottom_sleeve, 1)
            left_bottom_sleeve_mask = cv2.flip(right_bottom_sleeve_mask, 1)
            part_imgs[3] = left_bottom_sleeve
            part_clothes_masks[3] = left_bottom_sleeve_mask
        elif np.sum(right_bottom_sleeve_mask) == 0 and np.sum(left_bottom_sleeve_mask) > 0:
            left_bottom_sleeve = part_imgs[5]
            right_bottom_sleeve = cv2.flip(left_bottom_sleeve, 1)
            right_bottom_sleeve_mask = cv2.flip(left_bottom_sleeve_mask, 1)
            part_imgs[5] = right_bottom_sleeve
            part_clothes_masks[5] = right_bottom_sleeve_mask

        img = np.concatenate(part_imgs, axis = 2)
        img_lower = np.concatenate(part_imgs_lower, axis=2)

        return img, img_lower, denorm_upper_img, denorm_lower_img

    def __getitem__(self, idx):
        image, clothes, pose, clothes_pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, \
            retain_mask, skin_average, lower_label_map, lower_clothes_upper_bound, \
            person_name, clothes_name = self._load_raw_image(self._raw_idx[idx])

        image = image.transpose(2, 0, 1)                    # HWC => CHW
        clothes = clothes.transpose(2,0,1)
        pose = pose.transpose(2, 0, 1)                      # HWC => CHW
        clothes_pose = clothes_pose.transpose(2,0,1)
        norm_img = norm_img.transpose(2, 0, 1)
        norm_img_lower = norm_img_lower.transpose(2,0,1)
        denorm_upper_img = denorm_upper_img.transpose(2,0,1)
        denorm_lower_img = denorm_lower_img.transpose(2,0,1)
        denorm_upper_mask = (np.sum(denorm_upper_img, axis=0, keepdims=True)>0).astype(np.uint8)
        denorm_lower_mask = (np.sum(denorm_lower_img, axis=0, keepdims=True)>0).astype(np.uint8)

        skin_average = skin_average.transpose(2,0,1)
        retain_mask = retain_mask.transpose(2,0,1)
        lower_label_map = lower_label_map.transpose(2,0,1)
        lower_clothes_upper_bound = lower_clothes_upper_bound.transpose(2,0,1)

        return image.copy(), clothes.copy(), pose.copy(), clothes_pose.copy(), norm_img.copy(), norm_img_lower.copy(), \
               denorm_upper_img.copy(), denorm_lower_img.copy(), denorm_upper_mask.copy(), denorm_lower_mask.copy(), \
               retain_mask.copy(), skin_average.copy(), lower_label_map.copy(), lower_clothes_upper_bound.copy(),\
               person_name, clothes_name
