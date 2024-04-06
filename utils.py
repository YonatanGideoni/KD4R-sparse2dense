import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataloaders.dataloader import MyDataloader
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
from models import Decoder

cmap = plt.cm.viridis


def parse_command():
    model_names = ['resnet18', 'resnet50', 'densenet57']
    loss_names = ['l1', 'l2', 'l1-a']
    data_names = ['nyudepthv2', 'kitti', 'make3d']
    decoder_names = Decoder.names
    modality_names = MyDataloader.modality_names

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--data', metavar='DATA', default='make3d',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: make3d)')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv2', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: deconv2)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run (default: 25)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default 1e-3)')
    parser.add_argument('--train-size', default=None, type=int,
                        metavar='TS', help='training dataset size (default None)')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('--img-output-size', type=int, default=64,
                        help='Image output size')
    parser.add_argument('--output-channels', type=int, default=1,
                        help='# of output channels (default: 1)')
    parser.set_defaults(pretrained=False)

    args = parser.parse_args()

    return args


def save_checkpoint(state, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)


def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to half the initial LR every 5 epochs"""
    lr = lr_init * (0.5 ** (epoch // 5))
    print(f'Learning rate for epoch {epoch}: {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_output_directory(args):
    output_directory = os.path.join('results',
                                    '{}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}'.
                                    format(args.data, args.modality, args.arch, args.decoder, args.criterion,
                                           args.lr, args.batch_size))
    return output_directory


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    # todo - is this the correct way to do this? discrepancy between training objective and visualisation
    depth_target = interp_pred(depth_target, input.shape)

    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def interp_pred(pred: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    _, _, h, w = target_shape
    pred = torch.nn.functional.interpolate(pred,
                                           [h, w],
                                           mode='bilinear')

    return pred


def get_make3d_mask(target: torch.Tensor, MAX_DIST: float = 70.) -> torch.Tensor:
    return ((target > 0) & (target < MAX_DIST)).detach()
