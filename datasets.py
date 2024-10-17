# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from scipy.linalg import sqrtm
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils_HSI import open_file

DATASETS_CONFIG = {
        'PaviaC': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat', 
                     'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
            'img': 'Pavia.mat',
            'gt': 'Pavia_gt.mat'
            },
        'PaviaU': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'PaviaU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'KSC': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                     'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
            'img': 'KSC.mat',
            'gt': 'KSC_gt.mat'
            },
        'IndianPines': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                     'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
            'img': 'Indian_pines_corrected.mat',
            'gt': 'Indian_pines_gt.mat'
            },
        'Botswana': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                     'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
            'img': 'Botswana.mat',
            'gt': 'Botswana_gt.mat',
            },
        'Houston13': {
            'img': 'Houston13.mat',
            'gt': 'Houston13_7gt.mat',
            },
        'Houston18': {
            'img': 'Houston18.mat',
            'gt': 'Houston18_7gt.mat',
            },
        'YCZY': {
            'img': 'YC_ZY1_02D_data.mat.mat',
            'gt': 'YC_ZY1_02D_gt.mat',
            },
        'YCGF': {
            'img': 'YC_GF5_data.mat.mat',
            'gt': 'YC_GF5_gt.mat',
            },
        'YCZY9': {
            'img': 'YC_ZY1_02D_data.mat.mat',
            'gt': 'YC_ZY9_gt.mat',
            },
        'HHK': {
            'img': 'HHK_ZY_data.mat.mat',
            'gt': 'HHK_ZY_gt.mat',
            },
        'YCZY8': {
            'img': 'YC_ZY1_02D_data.mat.mat',
            'gt': 'YC_ZY_qj8_gt.mat',
            },
        'HHK8': {
            'img': 'HHK_ZY_data.mat.mat',
            'gt': 'HHK_ZY_qj8_gt.mat',
            },
        'Shanghai': {
            'img': 'Shanghai.mat',
            'gt': 'Shanghai_gt.mat',
            },
        'Hangzhou': {
            'img': 'Hangzhou.mat',
            'gt': 'Hangzhou_gt.mat',
            },
        'Dioni': {
            'img': 'Dioni.mat',
            'gt': 'Dioni_gt_out68.mat',
            },
        'Loukia': {
            'img': 'Loukia.mat',
            'gt': 'Loukia_gt_out68.mat',
            },
    }

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder# + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                     reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')['pavia'][:, :, :96]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaC_gt.mat')['PaviaC_gt']
        label_values = ["Undefined", "Asphalt", "Meadows", "Trees", "Bare Soil", "Bitumen",
                        "Self-Blocking Bricks", "Shadows"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU'][:, :, :96]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaUnew_gt.mat')['PaviaU_gt']

        label_values = ["Undefined", "Asphalt", "Meadows", "Trees", "Bare Soil", "Bitumen",
                        "Self-Blocking Bricks", "Shadows"]

        ignored_labels = [0]

    elif dataset_name == 'YCZY':
        # Load the image
        img = open_file(folder + 'ZY_YC_147_data.mat')['Data1']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'YC_ZY1_02D_gt.mat')['DataClass']

        label_values = ["Undefined", "1", " 3 ", "5", "6", "7",
                        "9", "13"]

        ignored_labels = [0]

    elif dataset_name == 'YCGF':
        # Load the image
        img = open_file(folder + 'YC_GF5_data.mat')['Data2']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'YC_GF5_gt.mat')['DataClass']

        label_values = ["Undefined", "14", " 11 ", "13", "9", "15",
                        "1", "2"]

        ignored_labels = [0]
    elif dataset_name == 'YCZY9':
        # Load the image
        img = open_file(folder + 'YC_ZY1_02D_data.mat')['X_YC'][:, :, :96]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'YC_ZY_8luo_gt.mat')['DataClass']

        label_values = ["Undefined", "1", " 2 ", "3", "6", "7",
                        "9", "11", " 12 ", "15"]
        # 建筑,河流,裸地,水田,闲置耕地,鱼塘,海洋,盐池,互花米草
        ignored_labels = [0]
    elif dataset_name == 'HHK':
        # Load the image
        img = open_file(folder + 'HHK_ZY_data.mat')['X_HHK'][:, :, :96]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'HHK_quluo8_gt.mat')['DataClass3']

        label_values = ["Undefined", "18", " 8 ", "9", "20", "13",
                        "10", "7 3 4", " 1 ", "2"]
        # 建筑,河流,裸地,水田,闲置耕地,鱼塘,海洋,san盐田,互花米草
        ignored_labels = [0]
    elif dataset_name == 'YCZY8':
        # Load the image
        img = open_file(folder + 'YC_ZY1_02D_data.mat')['X_YC'][:, :, :96]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'YC_ZY_qj8_gt.mat')['DataClass']

        label_values = ["Undefined", "1", " 2 ", "3", "6", "7",
                        "9", "11", " 12 ", "15"]

        ignored_labels = [0]
    elif dataset_name == 'HHK8':
        # Load the image
        img = open_file(folder + 'HHK_ZY_data.mat')['X_HHK'][:, :, :96]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'ZY_HHK_qj8_gt.mat')['DataClass3']

        label_values = ["Undefined", "18", " 8 ", "9", "20", "13",
                        "10", "7 3 4", " 1 ", "2"]

        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = open_file(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    elif dataset_name == 'Houston13':
        # Load the image
        img = open_file(folder + 'Houston13.mat')['ori_data']

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Houston13_7gt.mat')['map']
        # Q = open_file(folder + 'DMLSR_13to18.mat')['Q']

        label_values = ['Grass healthy','Grass stressed','Trees','Water','Residential buildings',
                        'Non-residential buildings','Road']

        ignored_labels = [0]
    elif dataset_name == 'Houston18':
        # Load the image
        img = open_file(folder + 'Houston18.mat')['ori_data']

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Houston18_7gt.mat')['map']
        # Q = open_file(folder + 'DMLSR_13to18.mat')['Q']

        label_values = ['Grass healthy','Grass stressed','Trees','Water','Residential buildings',
                        'Non-residential buildings','Road']

        ignored_labels = [0]
    elif dataset_name == 'Shanghai':
        # Load the image
        img = open_file(folder + 'Shanghai.mat')['ori_data'][:, :, :192]

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Shanghai_gt.mat')['map']

        label_values = ['Water','Land/Building','Plant']

        ignored_labels = [0]
    elif dataset_name == 'Hangzhou':
        # Load the image
        img = open_file(folder + 'Hangzhou.mat')['ori_data'][:, :, :192]

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Hangzhou_gt.mat')['map']

        label_values = ['Water','Land/Building','Plant']

        ignored_labels = [0]
    elif dataset_name == 'Dioni':
        # Load the image
        img = open_file(folder + 'Dioni.mat')['ori_data'][:, :, :144]

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Dioni_gt_out68.mat')['map']

        label_values = ["1", "2", "3",
                        "4", "5",
                        "6", "7",
                        "8", "9", "10",
                        "11", "12"]

        ignored_labels = [0]
    elif dataset_name == 'Loukia':
        # Load the image
        img = open_file(folder + 'Loukia.mat')['ori_data'][:, :, :144]

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Loukia_gt_out68.mat')['map']

        label_values = ["1", "2", "3",
                        "4", "5",
                        "6", "7",
                        "8", "9", "10",
                        "11", "12"]
                        
        ignored_labels = [0]
    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # img_temp = img.sum(2)
    # img_temp[img_temp==0]=1
    # img = img/np.expand_dims(img_temp,axis=2).min()
    
    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img= img.reshape((m*n,-1))
    img = img/img.max()
    img_temp = np.sqrt(np.asarray((img**2).sum(1)))
    img_temp = np.expand_dims(img_temp,axis=1)
    # img_temp = np.expand_dims(np.asarray(img_temp.sum(1)),axis=1)
    img_temp = img_temp.repeat(d,axis=1)
    img_temp[img_temp==0]=1
    img = img/img_temp
    # img = np.reshape(np.dot(img,Q),(m,n,-1))
    img = np.reshape(img,(m,n,-1))

    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        # self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        
        # state = np.random.get_state()
        # np.random.shuffle(self.indices)
        # np.random.set_state(state)
        # np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
                data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        return data, label

    # def __getitem__(self, i):
    #     x, y = self.indices[i]
    #     x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
    #     x2, y2 = x1 + self.patch_size, y1 + self.patch_size

    #     data = self.data[x1:x2, y1:y2]
    #     label = self.label[x1:x2, y1:y2]
    #     if self.transform is not None:
    #         data = self.transform(data) 
    #     return data, self.labels[i]

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data, self.label = next(self.loader)

        except StopIteration:
            self.next_input = None

            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        label = self.label

        self.preload()
        return data, label