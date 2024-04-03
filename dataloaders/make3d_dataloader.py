# taken from https://github.com/ZM-Zhou/SMDE-Pytorch/

import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from PIL import Image, ImageFile
from scipy.io import loadmat
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_TRAIN_DIR = 'Train400Img'
IMG_TEST_DIR = 'Test134'
DEPTH_TRAIN_DIR = 'Train400Depth'
DEPTH_TEST_DIR = 'Gridlaserdata'


def get_input_img(path, color=True):
    """Read the image in KITTI."""
    img = Image.open(path)
    if color:
        img = img.convert('RGB')
    return img


def get_input_depth_make3d(path):
    m = loadmat(path)
    pos3dgrid = m['Position3DGrid']
    depth = pos3dgrid[:, :, 3]
    return depth


def read_calib_file(path):
    """Read KITTI calibration file (from https://github.com/hunse/kitti)"""
    float_chars = set('0123456789.e+- ')
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format (adapted from
    https://github.com/hunse/kitti)"""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices (adapted from
    https://github.com/nianticlabs/monodepth2)"""
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


class Make3DDataset(data.Dataset):
    DATA_NAME_DICT = {
        'color': (IMG_TRAIN_DIR, 'img-', 'jpg'),
        'depth': (DEPTH_TRAIN_DIR, 'depth_sph_corr-', 'mat')
    }

    def __init__(self,
                 dataset_dir: str,
                 train: bool = True,
                 normalize_params=[0.411, 0.432, 0.45],
                 use_godard_crop=True,
                 full_size=None,
                 resize_before_crop=False
                 ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.use_godard_crop = use_godard_crop
        self.full_size = full_size
        self.resize_before_crop = resize_before_crop

        if not train:
            self.DATA_NAME_DICT['color'] = (IMG_TEST_DIR, 'img-', 'jpg')
            self.DATA_NAME_DICT['depth'] = (DEPTH_TEST_DIR, 'depth_sph_corr-', 'mat')

        data_dir = os.path.join(self.dataset_dir, IMG_TRAIN_DIR if train else IMG_TEST_DIR)
        self.file_list = self._get_file_list(data_dir)

        # Initializate transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):
        file_info = self.file_list[f_idx]
        base_path = os.path.join(self.dataset_dir, '{}',
                                 '{}' + file_info + '.{}')
        inputs = {}
        color_l_path = base_path.format(*self.DATA_NAME_DICT['color'])
        inputs['color_s_raw'] = get_input_img(color_l_path)

        depth_path = base_path.format(*self.DATA_NAME_DICT['depth'])
        inputs['depth'] = get_input_depth_make3d(depth_path)

        for key in list(inputs):
            if 'color' in key:
                raw_img = inputs[key]
                if self.resize_before_crop:
                    self.color_resize = tf.Resize(self.full_size,
                                                  interpolation=Image.ANTIALIAS)

                    img = self.to_tensor(self.color_resize(raw_img))
                    if self.use_godard_crop:
                        top = int((self.full_size[0] - self.full_size[1] / 2) / 2) + 1
                        bottom = int((self.full_size[0] + self.full_size[1] / 2) / 2) + 1
                        img = img[:, top:bottom, :]
                    inputs[key.replace('_raw', '')] = \
                        self.normalize(img)
                else:
                    if self.use_godard_crop:
                        raw_img = raw_img.crop((0, 710, 1704, 1562))
                    img = self.to_tensor(raw_img)
                    if self.full_size is not None:
                        # for outputting the same image with cv2
                        img = img.unsqueeze(0)
                        img = F.interpolate(img, self.full_size, mode='nearest')
                        img = img.squeeze(0)
                    inputs[key.replace('_raw', '')] = \
                        self.normalize(img)

            elif 'depth' in key:
                raw_depth = inputs[key]
                if self.use_godard_crop:
                    raise NotImplementedError('what are these numbers?')
                    raw_depth = raw_depth[17:38, :]
                depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)
                inputs[key] = depth

        # delete raw data
        inputs.pop('color_s_raw')
        inputs['file_info'] = [file_info]
        return inputs

    def _get_file_list(self, data_dir):
        files = os.listdir(data_dir)
        filenames = []
        for f in files:
            if f[-3:] != 'jpg':
                continue

            file_name = f.replace('\n', '').replace('.jpg', '').replace('img-', '')
            filenames.append(file_name)
        return filenames
