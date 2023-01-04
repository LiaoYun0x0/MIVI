import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry import spatial_soft_argmax2d, spatial_expectation2d
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange, repeat

from common.functions import * 
from common.nest import NestedTensor
from models.loftr import LoFTRModule
from models.position import PositionEmbedding2D, PositionEmbedding1D
from models.transformer import LocalFeatureTransformer,GlobalFeatureTransformer,PositionEncodingSine
from models.networks import GLNet


# local transformer parameters
cfg={}
cfg["lo_cfg"] = {}
lo_cfg = cfg["lo_cfg"]
lo_cfg["d_model"] = 128
lo_cfg["layer_names"] = ["self","cross"] * 1
lo_cfg["nhead"] = 8
lo_cfg["attention"] = "linear"


def _transform_inv(img,mean,std):
    img = img * std + mean
    img  = np.uint8(img * 255.0)
    img = img.transpose(1,2,0)
    return img

class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                           padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        #self.elu = nn.ELU(inplace=True)
        self.mish = nn.Mish(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        #x = self.elu(x)
        x = self.mish(x)
        return x



class MatchingNet(nn.Module):
    def __init__(
        self,
        d_coarse_model: int=256,
        d_fine_model: int=128,
        n_coarse_layers: int=6,
        n_fine_layers: int=4,
        n_heads: int=8,
        backbone_name: str='resnet18',
        matching_name: str='sinkhorn',
        match_threshold: float=0.2,
        window: int=5,
        border: int=1,
        sinkhorn_iterations: int=50,
    ):
        super().__init__()

        self.backbone = GLNet(backbone="resnet50")
        self.position2d = PositionEmbedding2D(d_coarse_model)
        self.position1d = PositionEmbedding1D(d_fine_model, max_len=window**2)

        self.local_transformer = LocalFeatureTransformer(cfg["lo_cfg"])

        self.proj = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.merge = nn.Linear(d_coarse_model, d_fine_model, bias=True)

        #self.conv2d = nn.Conv2d(d_coarse_model, d_fine_model, 1, 1)
        self.convbn = ConvBN(d_coarse_model, d_fine_model, 1, 1)

        self.regression1 = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.regression2 = nn.Linear(3200, d_fine_model, bias=True)
        self.regression = nn.Linear(d_fine_model, 2, bias=True)
        self.dropout = nn.Dropout(0.5)

        #self.L2Normalize = lambda feat, dim: feat / torch.pow(torch.sum(torch.pow(feat, 2), dim=dim) + 1e-6, 0.5).unsqueeze(dim)


        self.border = border
        self.window = window
        self.num_iter = sinkhorn_iterations
        self.match_threshold = match_threshold
        self.matching_name = matching_name
        self.step_coarse = 8
        self.step_fine = 2

        if matching_name == 'sinkhorn':
            bin_score = nn.Parameter(torch.tensor(1.))
            self.register_parameter("bin_score", bin_score)
        self.th = 0.1

    def fine_matching(self,x0,x1):
        x0,x1 = self.local_transformer(x0,x1)
        #x0, x1 = self.L2Normalize(x0, dim=0), self.L2Normalize(x1, dim=0)
        return x0,x1


    def _regression(self, feat):
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)
        feat = self.dropout(feat)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat

    def compute_confidence_matrix(self, query_lf,refer_lf, gt_matrix=None):
        _d =  query_lf.shape[-1]
        query_lf = query_lf / _d
        refer_lf = refer_lf / _d
        similarity_matrix = torch.matmul(query_lf,refer_lf.transpose(1,2)) / 0.1
        #sim_matrix = torch.einsum("nlc,nsc->nls", query_lf, refer_lf) / 0.1
        confidence_matrix = torch.softmax(similarity_matrix,1) * torch.softmax(similarity_matrix,2)
        return confidence_matrix

    def unfold_within_window(self, featmap):
        scale = self.step_coarse - self.step_fine
        #stride = int(math.pow(2, scale))
        stride = 4

        featmap_unfold = F.unfold(
            featmap,
            kernel_size=(self.window, self.window),
            stride=stride,
            padding=self.window//2
        )

        featmap_unfold = rearrange(
            featmap_unfold,
            "B (C MM) L -> B L MM C",
            MM=self.window ** 2
        )
        return featmap_unfold


    def forward(self, samples0, samples1):
        
        device = samples0.device

        #1x1600x256, 1x256x160x160
        (mdesc0_8x, mdesc1_8x), (fine_featmap0_2x, fine_featmap1_2x), (mdesc0_16x, mdesc1_16x), (fine_featmap0_4x, fine_featmap1_4x) = self.backbone.forward_pair_lo(samples0, samples1)

        cm_matrix_8x = self.compute_confidence_matrix(mdesc0_8x, mdesc1_8x)
        #mask = cm_matrix > self.th
        cf_matrix_8x = cm_matrix_8x * (cm_matrix_8x == cm_matrix_8x.max(dim=2, keepdim=True)[0]) * (cm_matrix_8x == cm_matrix_8x.max(dim=1, keepdim=True)[0])
        mask_v, all_j_ids = cf_matrix_8x.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        matches_8x = torch.stack([b_ids, i_ids, j_ids]).T


        mkpts0_8x, mkpts1_8x = batch_get_mkpts( matches_8x, samples0, samples1)

        fine_featmap0_2x = self.convbn(fine_featmap0_2x)
        fine_featmap1_2x = self.convbn(fine_featmap1_2x)

        fine_featmap0_unfold = self.unfold_within_window(fine_featmap0_2x) # 1x1600x25x256
        fine_featmap1_unfold = self.unfold_within_window(fine_featmap1_2x)

        local_desc = torch.cat([
            fine_featmap0_unfold[matches_8x[:, 0], matches_8x[:, 1]],
            fine_featmap1_unfold[matches_8x[:, 0], matches_8x[:, 2]]
        ], dim=0)

        center_desc = repeat(torch.cat([
            mdesc0_8x[matches_8x[:, 0], matches_8x[:, 1]],
            mdesc1_8x[matches_8x[:, 0], matches_8x[:, 2]]
            ], dim=0), 
            'N C -> N WW C', 
            WW=self.window**2)

        center_desc = self.proj(center_desc)
        local_desc = torch.cat([local_desc, center_desc], dim=-1)
        local_desc = self.merge(local_desc)
        local_position = self.position1d(local_desc)
        local_desc = local_desc + local_position

        desc0, desc1 = torch.chunk(local_desc, 2, dim=0)
        fdesc0, fdesc1 = self.fine_matching(desc0, desc1)

        c = self.window ** 2 // 2

        center_desc = repeat(fdesc0[:, c, :], 'N C->N WW C', WW=self.window**2)
        center_desc = torch.cat([center_desc, fdesc1], dim=-1)

        expected_coords = self._regression(center_desc)
        
        mkpts1_8x = mkpts1_8x[:, 1:] + expected_coords
        #16v4

        cm_matrix_16x = self.compute_confidence_matrix(mdesc0_16x, mdesc1_16x)
        #mask = cm_matrix > self.th
        cf_matrix_16x = cm_matrix_16x * (cm_matrix_16x == cm_matrix_16x.max(dim=2, keepdim=True)[0]) * (cm_matrix_16x == cm_matrix_16x.max(dim=1, keepdim=True)[0])
        mask_v, all_j_ids = cf_matrix_16x.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        matches_16x = torch.stack([b_ids, i_ids, j_ids]).T


        mkpts0_16x, mkpts1_16x = batch_get_mkpts( matches_16x, samples0, samples1, patch_size=16)

        fine_featmap0_4x = self.convbn(fine_featmap0_4x)
        fine_featmap1_4x = self.convbn(fine_featmap1_4x)

        fine_featmap0_unfold = self.unfold_within_window(fine_featmap0_4x) # 1x1600x25x256
        fine_featmap1_unfold = self.unfold_within_window(fine_featmap1_4x)

        local_desc = torch.cat([
            fine_featmap0_unfold[matches_16x[:, 0], matches_16x[:, 1]],
            fine_featmap1_unfold[matches_16x[:, 0], matches_16x[:, 2]]
        ], dim=0)

        center_desc = repeat(torch.cat([
            mdesc0_16x[matches_16x[:, 0], matches_16x[:, 1]],
            mdesc1_16x[matches_16x[:, 0], matches_16x[:, 2]]
            ], dim=0), 
            'N C -> N WW C', 
            WW=self.window**2)

        center_desc = self.proj(center_desc)
        local_desc = torch.cat([local_desc, center_desc], dim=-1)
        local_desc = self.merge(local_desc)
        local_position = self.position1d(local_desc)
        local_desc = local_desc + local_position

        desc0, desc1 = torch.chunk(local_desc, 2, dim=0)
        fdesc0, fdesc1 = self.fine_matching(desc0, desc1)

        c = self.window ** 2 // 2

        center_desc = repeat(fdesc0[:, c, :], 'N C->N WW C', WW=self.window**2)
        center_desc = torch.cat([center_desc, fdesc1], dim=-1)

        expected_coords = self._regression(center_desc)
        
        mkpts1_16x = mkpts1_16x[:, 1:] + expected_coords


        return {
            'samples0': samples0,
            'samples1': samples1,
            'cm_matrix_8x': cm_matrix_8x,
            'matches_8x': matches_8x,
            'mkpts1_8x': mkpts1_8x,
            'mkpts0_8x': mkpts0_8x,
            'mdesc0_8x': mdesc0_8x,
            'mdesc1_8x': mdesc1_8x,
            'cm_matrix_16x': cm_matrix_16x,
            'matches_16x': matches_16x,
            'mkpts1_16x': mkpts1_16x,
            'mkpts0_16x': mkpts0_16x,
            'mdesc0_16x': mdesc0_16x,
            'mdesc1_16x': mdesc1_16x,
        }
