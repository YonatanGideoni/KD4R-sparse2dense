import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import interp_pred, get_make3d_mask


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class Make3DMaskedL1Loss(nn.Module):
    def __init__(self):
        super(Make3DMaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        pred = interp_pred(pred, target.shape)
        valid_mask = get_make3d_mask(target)

        diff = target - pred
        diff = diff[valid_mask]

        self.loss = diff.abs().mean()
        return self.loss


class Make3DMaskedAleatoricL1(nn.Module):
    def __init__(self):
        super(Make3DMaskedAleatoricL1, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        assert pred.shape[1] == 2, 'Aleatoric predictions should have 2 channels'

        pred_mean, pred_logdiversity = torch.chunk(pred, 2, dim=1)
        pred_diversity = torch.exp(pred_logdiversity)

        pred_mean = interp_pred(pred_mean, target.shape)
        pred_diversity = interp_pred(pred_diversity, target.shape)
        valid_mask = get_make3d_mask(target)

        loss = (target - pred_mean).abs() / pred_diversity + pred_diversity.log()
        loss = loss[valid_mask]

        self.loss = loss.mean()
        return self.loss


class Make3DMaskedMSELoss(nn.Module):
    def __init__(self):
        super(Make3DMaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        pred = interp_pred(pred, target.shape)
        valid_mask = get_make3d_mask(target)
        diff = target - pred
        diff = diff[valid_mask]

        self.loss = (diff ** 2).mean()
        return self.loss
