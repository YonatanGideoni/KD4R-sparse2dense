# taken from https://github.com/ZM-Zhou/SMDE-Pytorch/

import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from PIL import Image, ImageFile
from scipy.io import loadmat, savemat
from torch import nn
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

    def __init__(self,
                 dataset_dir: str,
                 train: bool = True,
                 normalize_params=[0.411, 0.432, 0.45],
                 full_size=(64, 64),
                 resize_before_crop=False,
                 teacher: nn.Module = None,
                 ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.full_size = full_size
        self.resize_before_crop = resize_before_crop

        self.DATA_NAME_DICT = {
            'color': (IMG_TRAIN_DIR, 'img-', 'jpg'),
            'depth': (DEPTH_TRAIN_DIR, 'depth_sph_corr-', 'mat'),
            'teacher': ('teacher', 'teachout-', 'mat')
        }
        if not train:
            self.DATA_NAME_DICT['color'] = (IMG_TEST_DIR, 'img-', 'jpg')
            self.DATA_NAME_DICT['depth'] = (DEPTH_TEST_DIR, 'depth_sph_corr-', 'mat')

        data_dir = os.path.join(self.dataset_dir, IMG_TRAIN_DIR if train else IMG_TEST_DIR)
        self.file_list = self._get_file_list(data_dir)

        # Initializate transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])

        self.teacher = teacher

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):
        file_info = self.file_list[f_idx]
        base_path = os.path.join(self.dataset_dir, '{}',
                                 '{}' + file_info + '.{}')
        color_l_path = base_path.format(*self.DATA_NAME_DICT['color'])
        img = get_input_img(color_l_path)

        depth_path = base_path.format(*self.DATA_NAME_DICT['depth'])
        depth = get_input_depth_make3d(depth_path)

        if self.resize_before_crop:
            self.color_resize = tf.Resize(self.full_size,
                                          interpolation=Image.ANTIALIAS)

            img = self.to_tensor(self.color_resize(img))
        else:
            img = self.to_tensor(img)
            if self.full_size is not None:
                # for outputting the same image with cv2
                img = img.unsqueeze(0)
                img = F.interpolate(img, self.full_size, mode='nearest')
                img = img.squeeze(0)
        input_img = self.normalize(img)

        depth = torch.from_numpy(depth.copy()).unsqueeze(0)

        teach_res = self.get_teach_output(base_path, input_img)
        if teach_res is not None:
            gt = torch.stack([depth, teach_res], dim=0)
            return input_img, gt

        return input_img, depth

    def _get_file_list(self, data_dir):
        files = os.listdir(data_dir)
        filenames = []
        for f in files:
            if f[-3:] != 'jpg':
                continue

            file_name = f.replace('\n', '').replace('.jpg', '').replace('img-', '')
            filenames.append(file_name)
        return filenames

    def get_teach_output(self, base_path, input_img):
        if self.teacher is None:
            return None

        teacher_path = base_path.format(*self.DATA_NAME_DICT['teacher'])
        if os.path.exists(teacher_path):
            teach_out = loadmat(teacher_path)['teach_out']
            return teach_out

        self.teacher.eval()
        with torch.no_grad():
            teach_out = self.teacher(input_img.unsqueeze(0)).squeeze(0)
        savemat(teacher_path, {'teach_out': teach_out})

        return teach_out
