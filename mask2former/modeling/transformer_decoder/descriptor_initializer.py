from einops import rearrange
from torch import Tensor
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mask2former.data.points.annotation_generator import create_circular_mask

# class AvgPoolingInitializer(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.fg_thresh = 0.5
#         self.bg_thresh = 0.5

#     def forward(self, fmap: Tensor, fg_mask: Tensor) -> Tuple[Tensor, Tensor]:
#         """
#         Forward method
#         :param fmap: tensor of shape [B, C, H, W]
#         :param fg_mask: tensor of shape [B, I, H, W] with values in [0,1]
#         # :param bg_mask: tensor of shape [B, Qb, H, W] with values in [0,1] 
#         :return: tuple of tensors of shape [B, I, C] and [B, Qb, C]
#         """
#         # bg_mask = self.unfold(bg_mask)  # [B, 1, grid_patch_size, Qb]
#         # fmap_unfolded = self.unfold(fmap)  # [B, C, grid_patch_size, Qb]

#         ch = fmap.size(1)
#         fmap = rearrange(fmap, "B C H W -> B (H W) C")
#         fg_mask = rearrange(fg_mask, "B I H W -> B I (H W)")
#         # bg_mask = rearrange(bg_mask, "B Qb H W -> B Qb (H W)")

#         # fg_init, bg_init = [], []
#         fg_init = []

#         # for fmap_ps, fg_mask_ps, bg_mask_ps in zip(fmap, fg_mask, bg_mask):
#         for fmap_ps, fg_mask_ps in zip(fmap, fg_mask):    

#             fg_init.append([])
#             for fg_mask_ps_i in fg_mask_ps:
#                 fg_pt_coords = (fg_mask_ps_i > self.fg_thresh).nonzero(as_tuple=False).squeeze(1)  # [N]

#                 if fg_pt_coords.numel() == 0:
#                     fg_pt_coords = fg_mask_ps_i.argmax()[None]  # [1]

#                 fg_init[-1].append(fmap_ps[fg_pt_coords].mean(0))

#             fg_init[-1] = torch.stack(fg_init[-1], 0)

#             # bg_init.append([])
#             # for bg_mask_ps_i in bg_mask_ps:
#             #     bg_pt_coords = (bg_mask_ps_i > self.bg_thresh).nonzero(as_tuple=False).squeeze(1)  # [N]

#             #     if bg_pt_coords.numel() == 0:
#             #         bg_init[-1].append(torch.zeros(ch, dtype=fmap.dtype, device=fmap.device))
#             #     else:
#             #         bg_init[-1].append(fmap_ps[bg_pt_coords].mean(0))

#             # bg_init[-1] = torch.stack(bg_init[-1], 0)

#         return torch.stack(fg_init, 0)
#         # return torch.stack(fg_init, 0), torch.stack(bg_init, 0)


class AvgClicksPoolingInitializer(nn.Module):

    def __init__(self, multi_scale: bool = True ):
        super().__init__()
        self.fg_thresh = 0.5
        self.bg_thresh = 0.5
        self.multi_scale = multi_scale

    def forward(self, features: Tensor, scribbles,  batched_fg_coords_list: list, batched_bg_coords_list: list,
                height: int=None, width: int=None, random_bg_queries: bool=False) -> Tensor:
        """
        Forward method
        :param fmaps: multiscale feature maps tensor of shape [B, C, H, W]
        :param scribbles: tensor of shape [B, I, H, W] with values in [0,1]
        :return: tuple of tensors of shape [B, I, C] and [B, Qb, C]
        """
        # bg_mask = self.unfold(bg_mask)  # [B, 1, grid_patch_size, Qb]
        # fmap_unfolded = self.unfold(fmap)  # [B, C, grid_patch_size, Qb]
        if random_bg_queries:
            if self.multi_scale:
                descriptors = []
                
        if random_bg_queries:
            if self.multi_scale:
                descriptors = []
                if scribbles is None:
                    for b_num, fg_coords_per_image in enumerate(batched_fg_coords_list):
                        feature_levels = len(features)
                        query_descriptors = []
                        point_masks = self._coords_to_point_masks(fg_coords_per_image,first_click_center=True, first_click_radius = 8,
                                height = height, width = width, device= features[0][0].device)
                        if  batched_bg_coords_list[b_num]:
                            bg_point_masks = self._coords_to_point_masks([batched_bg_coords_list[b_num]],first_click_center=False, first_click_radius = 8,
                                height = height, width = width, device= features[0][0].device)
                            point_masks = torch.cat((point_masks,bg_point_masks),dim=0)
                        point_masks = point_masks.unsqueeze(0)
                        for i in range(feature_levels):
                            fmap = features[i]
                            feat_b_num = fmap[b_num].unsqueeze(0)
                            
                            point_masks_resized = F.interpolate(point_masks, size=feat_b_num.shape[-2:], mode="bilinear", align_corners=False)
                            query_descriptors.append(self.get_descriptors(feat_b_num, point_masks_resized))
                        descriptors.append(torch.mean(torch.stack(query_descriptors, -1), dim = -1))
                    return descriptors
                else:
                    for b_num, scrbs in enumerate(scribbles):
                        feature_levels = len(features)
                        query_descriptors = []
                        scrbs = scrbs.unsqueeze(0)
                        for i in range(feature_levels):
                            fmap = features[i]
                            feat_b_num = fmap[b_num].unsqueeze(0)
                            
                            scrbs_resized = F.interpolate(scrbs.to(torch.float), size=feat_b_num.shape[-2:], mode="bilinear", align_corners=False)
                            query_descriptors.append(self.get_descriptors(feat_b_num,scrbs_resized))
                        descriptors.append(torch.mean(torch.stack(query_descriptors, -1), dim = -1))
                return descriptors
            # else:
            #     fmap = features[-1]
            #     descriptors = []
            #     for b_num, scrbs in enumerate(scribbles):
            #         scrbs = scrbs.unsqueeze(0)
            #         feat_b_num = fmap[b_num].unsqueeze(0)
            #         scrbs_resized = F.interpolate(scrbs, size=feat_b_num.shape[-2:], mode="bilinear", align_corners=False)
            #         descriptors.append(self.get_descriptors(feat_b_num,scrbs_resized))
            #     return descriptors

    def _coords_to_point_masks(self, point_coords_per_image, first_click_center=True, first_click_radius = 8,
                               height = None, width = None, device= None
    ):
        assert height is not None
        assert width is not None
        point_masks = []
        for coords_per_obj in point_coords_per_image:
            for i, coords in enumerate(coords_per_obj):
                if i==0 and first_click_center:
                    _pm = create_circular_mask(height, width, centers=[coords], radius=first_click_radius)
                else:
                    _pm = create_circular_mask(height, width, centers=[coords], radius=3)
                point_masks.append(_pm)
        point_masks = torch.from_numpy(np.stack(point_masks,axis=0)).to(device=device, dtype=torch.float)
        return point_masks

    def get_descriptors(self, fmap, fg_mask):

        fmap = rearrange(fmap, "B C H W -> B (H W) C")
        fg_mask = rearrange(fg_mask, "B I H W -> B I (H W)")
        # bg_mask = rearrange(bg_mask, "B Qb H W -> B Qb (H W)")

        # fg_init, bg_init = [], []
        fg_init = []

        # for fmap_ps, fg_mask_ps, bg_mask_ps in zip(fmap, fg_mask, bg_mask):
        for fmap_ps, fg_mask_ps in zip(fmap, fg_mask):    

            fg_init.append([])
            for fg_mask_ps_i in fg_mask_ps:
                fg_pt_coords = (fg_mask_ps_i > self.fg_thresh).nonzero(as_tuple=False).squeeze(1)  # [N]

                if fg_pt_coords.numel() == 0:
                    fg_pt_coords = fg_mask_ps_i.argmax()[None]  # [1]

                fg_init[-1].append(fmap_ps[fg_pt_coords].mean(0))

            fg_init[-1] = torch.stack(fg_init[-1], 0)

        return torch.stack(fg_init, 0)

class AvgPoolingInitializer(nn.Module):

    def __init__(self, multi_scale: bool = True ):
        super().__init__()
        self.fg_thresh = 0.5
        self.bg_thresh = 0.5
        self.multi_scale = multi_scale

    def forward(self, features: Tensor, scribbles: Tensor, random_bg_queries: bool=False) -> Tensor:
        """
        Forward method
        :param fmaps: multiscale feature maps tensor of shape [B, C, H, W]
        :param scribbles: tensor of shape [B, I, H, W] with values in [0,1]
        :return: tuple of tensors of shape [B, I, C] and [B, Qb, C]
        """
        # bg_mask = self.unfold(bg_mask)  # [B, 1, grid_patch_size, Qb]
        # fmap_unfolded = self.unfold(fmap)  # [B, C, grid_patch_size, Qb]

        if random_bg_queries:
            if self.multi_scale:
                descriptors = []
                for b_num, scrbs in enumerate(scribbles):
                    feature_levels = len(features)
                    query_descriptors = []
                    scrbs = scrbs.unsqueeze(0)
                    for i in range(feature_levels):
                        fmap = features[i]
                        feat_b_num = fmap[b_num].unsqueeze(0)
                        # print(scrbs.shape)
                        # print(feat_b_num.shape)
                        scrbs_resized = F.interpolate(scrbs, size=feat_b_num.shape[-2:], mode="bilinear", align_corners=False)
                        query_descriptors.append(self.get_descriptors(feat_b_num,scrbs_resized))
                    descriptors.append(torch.mean(torch.stack(query_descriptors, -1), dim = -1))
                return descriptors
            else:
                fmap = features[-1]
                descriptors = []
                for b_num, scrbs in enumerate(scribbles):
                    scrbs = scrbs.unsqueeze(0)
                    feat_b_num = fmap[b_num].unsqueeze(0)
                    scrbs_resized = F.interpolate(scrbs, size=feat_b_num.shape[-2:], mode="bilinear", align_corners=False)
                    descriptors.append(self.get_descriptors(feat_b_num,scrbs_resized))
                return descriptors
        else:
            if self.multi_scale:
                feature_levels = len(features)
                query_descriptors = []
                for i in range(feature_levels):
                    fmap = features[i]
                    scribbles_resized = F.interpolate(scribbles, size=fmap.shape[-2:], mode="bilinear", align_corners=False)
                    query_descriptors.append(self.get_descriptors(fmap,scribbles_resized))
                return torch.mean(torch.stack(query_descriptors, -1), dim = -1)
            else:
                fmap = features[-1]
                scribbles_resized = F.interpolate(scribbles, size=fmap.shape[-2:], mode="bilinear", align_corners=False)
                return self.get_descriptors(fmap,scribbles_resized)

    def get_descriptors(self, fmap, fg_mask):

        fmap = rearrange(fmap, "B C H W -> B (H W) C")
        fg_mask = rearrange(fg_mask, "B I H W -> B I (H W)")
        # bg_mask = rearrange(bg_mask, "B Qb H W -> B Qb (H W)")

        # fg_init, bg_init = [], []
        fg_init = []

        # for fmap_ps, fg_mask_ps, bg_mask_ps in zip(fmap, fg_mask, bg_mask):
        for fmap_ps, fg_mask_ps in zip(fmap, fg_mask):    

            fg_init.append([])
            for fg_mask_ps_i in fg_mask_ps:
                fg_pt_coords = (fg_mask_ps_i > self.fg_thresh).nonzero(as_tuple=False).squeeze(1)  # [N]

                if fg_pt_coords.numel() == 0:
                    fg_pt_coords = fg_mask_ps_i.argmax()[None]  # [1]

                fg_init[-1].append(fmap_ps[fg_pt_coords].mean(0))

            fg_init[-1] = torch.stack(fg_init[-1], 0)

        return torch.stack(fg_init, 0)