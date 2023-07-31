from einops import rearrange
from torch import Tensor
from typing import Tuple, Dict, Any
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamite.data.points.annotation_generator import create_circular_mask
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

class AvgClicksPoolingInitializer(nn.Module):

    def __init__(self, multi_scale: bool = True,  use_coords_on_point_mask: bool=True,
                use_point_features: bool=False):
        super().__init__()
        self.fg_thresh = 0.5
        self.bg_thresh = 0.5
        self.multi_scale = multi_scale
        self.use_coords_on_point_mask = use_coords_on_point_mask
        self.use_point_features = use_point_features

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
                if self.use_point_features:
                    for b_num, fg_coords_per_image in enumerate(batched_fg_coords_list):
                        feature_levels = len(features)
                        query_descriptors = []
                        
                        point_coords_per_image = copy.deepcopy(fg_coords_per_image[0])
                        for p in fg_coords_per_image[1:]:
                            point_coords_per_image.extend(p)
                        if  batched_bg_coords_list[b_num]:
                            point_coords_per_image.extend(copy.deepcopy(batched_bg_coords_list[b_num]))
                        
                        point_coords_per_image = torch.tensor(point_coords_per_image,dtype=torch.float, device=features[0][0].device)
                        point_coords_per_image = point_coords_per_image[:,:2]
                        _,_,h,w = features[-1].shape
                        H = float(h*8)
                        W = float(w*8)
                        point_coords_per_image[:,0]/=H
                        point_coords_per_image[:,1]/=W
                        point_coords_per_image = point_coords_per_image.flip(-1)
                        for i in range(feature_levels):
                            fmap = features[i]
                            feat_b_num = fmap[b_num].unsqueeze(0)

                            query_descriptors.append(self.get_features_descriptors(feat_b_num, point_coords_per_image.unsqueeze(0)))
                        descriptors.append(torch.mean(torch.stack(query_descriptors, -1), dim = -1))
                    return descriptors
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
                            
                            if self.use_coords_on_point_mask:
                                query_descriptors.append(self.get_descriptors_for_points(feat_b_num,scrbs))
                            else:
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
    
    def get_descriptors_for_points(self, fmap, point_masks):

        # point_masks: 1xQxHxW
        # fmap: 1xCxHxW        
        _,_,H,W = point_masks.shape
        
        fg_init = []
        for mask_i in point_masks.squeeze(0):
            points_coords_on_mask = torch.stack(torch.where(mask_i),dim=1).to(dtype=torch.float)
            points_coords_on_mask[:,0]/=float(H)
            points_coords_on_mask[:,1]/=float(W)
            points_coords_on_mask = points_coords_on_mask.flip(-1)
            y = point_sample(fmap, points_coords_on_mask.unsqueeze(0),align_corners=False) # 1xCxPoints
            fg_init.append(y.mean(2).squeeze(0))
        
        return torch.stack(fg_init, 0).unsqueeze(0)

    def get_features_descriptors(self, fmap, point_coords_per_image):

        # fmap: 1xCxHxW 
        # point_coords_per_image: 1XQx2   

        y = point_sample(fmap, point_coords_per_image,align_corners=False) # 1xCxPoints
        
        return torch.permute(y, (0, 2, 1))

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