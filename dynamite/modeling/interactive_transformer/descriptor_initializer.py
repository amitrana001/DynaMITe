
from torch import Tensor
from typing import Tuple, Dict, Any
import copy
import numpy as np
import torch
import torch.nn as nn
from detectron2.projects.point_rend.point_features import point_sample

class AvgClicksPoolingInitializer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, features: Tensor, batched_fg_coords_list: list, batched_bg_coords_list: list) -> Tensor:
        """
        Forward method
        :param fmaps: multiscale feature maps tensor of shape [B, C, H, W]
        :param scribbles: tensor of shape [B, I, H, W] with values in [0,1]
        :return: tuple of tensors of shape [B, I, C] and [B, Qb, C]
        """
        # bg_mask = self.unfold(bg_mask)  # [B, 1, grid_patch_size, Qb]
        # fmap_unfolded = self.unfold(fmap)  # [B, C, grid_patch_size, Qb]
                
        descriptors = []
            
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
          
    def get_features_descriptors(self, fmap, point_coords_per_image):

        # fmap: 1xCxHxW 
        # point_coords_per_image: 1XQx2   

        y = point_sample(fmap, point_coords_per_image,align_corners=False) # 1xCxPoints
        
        return torch.permute(y, (0, 2, 1))
