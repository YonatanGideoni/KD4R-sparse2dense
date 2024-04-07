import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import interp_pred, get_dist_mask


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, *args):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, *args):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class Make3DMaskedL1Loss(nn.Module):
    def __init__(self):
        super(Make3DMaskedL1Loss, self).__init__()

    def forward(self, pred, target, *args):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        pred = interp_pred(pred, target.shape)
        valid_mask = get_dist_mask(target)

        diff = target - pred
        diff = diff[valid_mask]

        self.loss = diff.abs().mean()
        return self.loss


class Make3DMaskedAleatoricL1(nn.Module):
    def __init__(self):
        super(Make3DMaskedAleatoricL1, self).__init__()

    def forward(self, pred, target, *args):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        assert pred.shape[1] == 2, 'Aleatoric predictions should have 2 channels'

        pred_mean, pred_logdiversity = torch.chunk(pred, 2, dim=1)
        pred_diversity = torch.exp(pred_logdiversity)

        pred_mean = interp_pred(pred_mean, target.shape)
        pred_diversity = interp_pred(pred_diversity, target.shape)
        valid_mask = get_dist_mask(target)

        loss = (target - pred_mean).abs() / pred_diversity + pred_diversity.log()
        loss = loss[valid_mask]

        self.loss = loss.mean()
        return self.loss


class Make3DMaskedMSELoss(nn.Module):
    def __init__(self):
        super(Make3DMaskedMSELoss, self).__init__()

    def forward(self, pred, target, *args):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        pred = interp_pred(pred, target.shape)
        valid_mask = get_dist_mask(target)
        diff = target - pred
        diff = diff[valid_mask]

        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedDistillationLossL1(nn.Module):
    def __init__(self, alpha=0.5):
        super(MaskedDistillationLossL1, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target, teacher_pred):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        assert teacher_pred.shape[1] == 1, 'Teacher predictions should have 1 channel for regular l1 loss'

        target_mask = get_dist_mask(target)
        orig_pred = pred
        pred = interp_pred(pred, target.shape)

        target_diff = target - pred
        target_diff = target_diff[target_mask]
        teacher_diff = teacher_pred - orig_pred

        self.loss = self.alpha * target_diff.abs().mean() + (1 - self.alpha) * teacher_diff.abs().mean()
        return self.loss


class MaskedDistillationLossAleatoricL1(nn.Module):
    def __init__(self, alpha=0.5):
        super(MaskedDistillationLossAleatoricL1, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target, teacher_pred):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        assert teacher_pred.shape[1] == 2, 'Teacher predictions should have 2 channels for aleatoric loss'

        target_mask = get_dist_mask(target)
        orig_pred_mean, orig_pred_logdiversity = torch.chunk(pred, 2, dim=1)
        pred_mean = interp_pred(orig_pred_mean, target.shape)
        orig_pred_diversity = torch.exp(orig_pred_logdiversity)
        pred_diversity = interp_pred(orig_pred_diversity, target.shape)

        target_diff = target - pred_mean
        target_diff = target_diff[target_mask]
        teacher_mean, teacher_logdiversity = torch.chunk(teacher_pred, 2, dim=1)
        teacher_diff = teacher_mean - orig_pred_mean
        teacher_diversity = torch.exp(teacher_logdiversity)

        loss = (target_diff.abs() / pred_diversity + pred_diversity.log()).mean()
        distill_loss = (teacher_diff.abs() / teacher_diversity +
                        teacher_logdiversity - orig_pred_logdiversity +
                        orig_pred_diversity / teacher_diversity).mean()

        self.loss = self.alpha * loss + (1 - self.alpha) * distill_loss
        return self.loss
