import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.functions import * 
# import gemo.homographies as homo
from common import Image
from common import NoGradientError

class MatchingCriterion(nn.Module):
    def __init__(
        self, 
        data_name: str,
        match_type: str='dual_doftmax',
        dist_thresh: float=5,
        weights: list=[1., 1000.], 
        eps=1e-10
    ):
        super().__init__()

        self.data_name = data_name
        self.match_type = match_type
        self.dist_thresh = dist_thresh
        self.ws = weights
        self.eps = eps
    
    def set_weight(self, std, mask=None, regularizer=0.):
        inverse_std = 1. / torch.clamp(std + regularizer, min=self.eps)
        weight = inverse_std / torch.mean(inverse_std)
        weight = weight.detach()

        if mask is not None:
            weight = weight.masked_select(mask.bool())
            weight /= (torch.mean(weight) + self.eps)
        
        return weight

    def coarse_loss(self, preds, targets):
        confidence_matrix = preds['cm_matrix']
        gt_matrix = targets['gt_matrix']
        loss = (-gt_matrix * torch.log(confidence_matrix + 1e-6)).sum() / preds['cm_matrix'].shape[0]
        return loss
    
    def compute_dist_within_images(
        self, mkpts0, mkpts1, image0: Image, image1: Image
    ):
        mkpts0_r = image1.project(image0.unproject(mkpts0.T)).T
        dist = torch.norm(mkpts1, mkpts0, dim=-1)

        return dist
    
    def fine_loss(self, preds, targets):

        samples0, samples1 = preds['samples0'], preds['samples1']
        mkpts0, mkpts1 = preds['mkpts0'], preds['mkpts1']

        gt_mask = targets['gt_matrix'] > 0
        gt_mask_v, gt_all_j_ids = gt_mask.max(dim=2)
        b_ids, i_ids = torch.where(gt_mask_v)
        j_ids = gt_all_j_ids[b_ids, i_ids]
        gt_matches = torch.stack([b_ids, i_ids, j_ids]).T

        gt_mkpts0, gt_mkpts1 = batch_get_mkpts(gt_matches, samples0, samples1)

        if gt_mkpts0.shape[0] == 0:
            return torch.tensor(1e-6, requires_grad=True).cuda()

        gt_mkpts11 = []
        mkpts11 = []
        mkpts10 = []
        gt_mkpts10 = []
        exp = []

        for idx, mkp in enumerate(mkpts0):
            m_gt_mkpts1 = gt_mkpts1[torch.where((gt_mkpts0==mkp).all(1))]
            #m_gt_mkpts0 = gt_mkpts0[torch.where((gt_mkpts0==mkp).all(1))]
            if m_gt_mkpts1.shape[0]!=0 and m_gt_mkpts1[0][0]==mkp[0]:
                if torch.norm(mkpts1[idx] - m_gt_mkpts1.squeeze()[1:]) < 10:
                    gt_mkpts11.append(m_gt_mkpts1.squeeze()[1:])
                    #gt_mkpts10.append(m_gt_mkpts0.squeeze()[1:])
                    mkpts11.append(mkpts1[idx])
                    #mkpts10.append(mkpts0[idx])
                    #exp.append(preds['exp'][idx])

        if len(gt_mkpts11) != 0:
            gt_mkpts11 = torch.stack(gt_mkpts11)
            #gt_mkpts10 = torch.stack(gt_mkpts10)
            mkpts11 = torch.stack(mkpts11)
            #mkpts10 = torch.stack(mkpts10)
            #exps = torch.stack(exp)
        else:
            gt_mkpts11 = torch.Tensor(0,2).cuda()
            mkpts11 = torch.Tensor(0,2).cuda()

        if len(gt_mkpts11) == 0:
            return torch.tensor(10., requires_grad=True).cuda()
        loss = torch.mean(torch.norm(gt_mkpts11 - mkpts11, p=2, dim=1))
        return torch.tensor(loss, requires_grad=True).cuda()

    def forward(self, preds, targets):

        coarse_loss = self.coarse_loss(
                preds, targets)
        fine_loss = self.fine_loss(preds, targets)
        #import pdb
        #pdb.set_trace()
        #print(coarse_loss, fine_loss)
        losses = self.ws[0] * coarse_loss + self.ws[1] * fine_loss
        loss_dict = {
            'losses': losses,
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss
        }
        #print(loss_dict)
        return loss_dict
