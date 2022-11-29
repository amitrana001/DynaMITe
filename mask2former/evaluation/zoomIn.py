import datetime
import logging
import time
import os
from contextlib import ExitStack, contextmanager
from traceback import walk_tb
from typing import List, Union
import torch
import torchvision
from torch import nn
import cv2
from detectron2.data import transforms as T
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.structures import BitMasks
from torch.nn import functional as F
import numpy as np
from detectron2.modeling.postprocessing import sem_seg_postprocess

class zoomIn:

    def __init__(self, cfg, expand_ratio=1.4):
        self.expand_ratio = expand_ratio

    def apply_zoom(self, inputs, outputs, images, scribbles, mask_features, multi_scale_features):

        # 1xQxHxW -> 1xHxW
        _,_,H,W = outputs['pred_masks'].shape
        pred_masks = outputs['pred_masks'][:,0,:] > 0
        pred_masks = pred_masks.to(dtype=torch.uint8)
        bboxes =  BitMasks(pred_masks).get_bounding_boxes().tensor #(x1, y1, x2, y2)
        bbox = bboxes[0]

        rmin, rmax, cmin, cmax = self.expand_bbox(bbox,self.expand_ratio)
        rmax = min(rmax, H)
        cmax = min(cmax, W)
        rmin = max(0, rmin)
        cmin = max(0, cmin)
        #to get original image box after removing padding and all
        H_orig, W_orig = inputs[0]['image'].shape[-2:]
        H_pad, W_pad = images.image_sizes[0][-2:]
        # print(H_orig,W_orig)
        # print(images.image_sizes[0])
        # print(H,W)
        dummy_mask = torch.zeros((1,1,H_pad,W_pad),dtype=torch.float)
        h_scale = H_pad/H
        w_scale = W_pad/W
        # print(h_scale,w_scale)
        s_rmin = int(round(rmin*h_scale))
        s_rmax= int(round(rmax*h_scale))
        s_cmin = int(round(cmin*w_scale))
        s_cmax = int(round(cmax*w_scale))
        dummy_mask[:,:,s_rmin:s_rmax+1, s_cmin:s_cmax+1]  = 1

        dummy_mask = sem_seg_postprocess(dummy_mask, images.image_sizes[0], H_orig, W_orig)
        # dummy_mask = dummy_mask
        # print(dummy_mask.shape)
        full_boxes =  BitMasks(dummy_mask).get_bounding_boxes().tensor #(x1, y1, x2, y2)
        full_bbox = np.asarray(full_boxes[0]).astype(np.int16)
        f_cmin, f_rmin, f_cmax, f_rmax = full_bbox
        full_box  = [f_rmin,f_rmax, f_cmin,f_cmax]
        #_____________bbox________________
        # bbox = [cmin, rmin, cmax+1, rmax+1]

        #(rmin,cmin)########
        #           #      #
        #           #      #
        #           #      #
        #           ########(rmax, cmax)   

        #_________________________________


        # for i, bbox in enumerate(bboxes):
       
        # self.inputs[0]["image"] = self.inputs[0]["image"][:,rmin:rmax+1, cmin:cmax+1]
        # self.inputs[0]['fg_scrbs'] = self.inputs[0]["fg_scrbs"][:,rmin:rmax+1, cmin:cmax+1]
        # self.inputs[0]['bg_scrbs'] = self.inputs[0]["bg_scrbs"][:,rmin:rmax+1, cmin:cmax+1]
        # self.inputs[0]["height"] = rmax-rmin+1
        # self.inputs[0]["width"] = cmax -cmin+1

        crop_mask_features = mask_features[:,:,rmin:rmax+1, cmin:cmax+1]
        crop_mask_features = F.interpolate(
            crop_mask_features,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        crop_multi_scale_features = []
        for feat in multi_scale_features:
            _,_,H_f,W_f =  feat.shape
            scaled_feat = F.interpolate(
                feat,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            crop_feat = scaled_feat[:,:,rmin:rmax+1, cmin:cmax+1]
            crop_feat = F.interpolate(
                crop_feat,
                size=(H_f, W_f),
                mode="bilinear",
                align_corners=False,
            )
            crop_multi_scale_features.append(crop_feat)
        
        _, H_s, W_s = scribbles[0].shape
        crop_scribbles = []
        for scrbs in scribbles:
            crop_scrbs = scrbs[:,rmin:rmax+1, cmin:cmax+1]
            crop_scrbs = F.interpolate(
                crop_scrbs.unsqueeze(0),
                size=(H_s, W_s),
                mode="bilinear",
                align_corners=False,
            )
            crop_scribbles.append(crop_scrbs.squeeze(0))
        return crop_scribbles, crop_mask_features, crop_multi_scale_features, [rmin,rmax,cmin,cmax], full_box
        # image = self.inputs[0]["image"][:,rmin:rmax+1, cmin:cmax+1].permute(1,2,0)
        # p_srb = self.inputs[0]["fg_scrbs"][:,rmin:rmax+1, cmin:cmax+1].squeeze(0)
        # p_srb = np.asarray(p_srb)
        

        # inputs = {}
        
        # h,w = image.shape[:2]
        # inputs["height"] = h
        # inputs["width"] = w
        # img=  self.transforms.get_transform(np.asarray(image)).apply_image(np.asarray(image))
        # inputs["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # fg_scrbs = self.transforms.get_transform(p_srb).apply_segmentation(p_srb)
        # bg_scrbs = None
        # if self.inputs[0]["bg_scrbs"] is not None:
        #     n_scrbs = self.inputs[0]["bg_scrbs"][:,rmin:rmax+1, cmin:cmax+1]
        #     n_scrbs = np.asarray(n_scrbs)
        #     bg_scrbs = []
        #     for n_scrb in n_scrbs:
        #         bg_scrbs.append(torch.from_numpy(self.transforms.get_transform(n_scrb).apply_segmentation(n_scrb)))
        #     bg_scrbs = torch.stack(bg_scrbs,0).float()
        # fg_scrbs = torch.from_numpy(fg_scrbs).unsqueeze(0).float()
        # # bg_scrbs = torch.from_numpy(bg_scrbs).unsqueeze(0).float()
        # inputs["fg_scrbs"] = fg_scrbs
        # inputs["bg_scrbs"] = bg_scrbs
        # inputs["scrbs_count"] = 1
        # # processed_results = model([inputs])[0][0]
        # # new_pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
        # # new_pred_masks = torchvision.transforms.Resize(size = (h,w))(pred_masks)
        # return inputs, [rmin,rmax,cmin,cmax]
    
    def expand_bbox(self, bbox, expand_ratio=1.4, min_crop_size=None):
        cmin, rmin, cmax, rmax = np.asarray(bbox)
        # rmin, rmax, cmin, cmax = bbox
        cmax-=1
        cmin-=1
        rcenter = 0.5 * (rmin + rmax)
        ccenter = 0.5 * (cmin + cmax)
        height = expand_ratio * (rmax - rmin + 1)
        width = expand_ratio * (cmax - cmin + 1)
        if min_crop_size is not None:
            height = max(height, min_crop_size)
            width = max(width, min_crop_size)

        rmin = int(round(rcenter - 0.5 * height))
        rmax = int(round(rcenter + 0.5 * height))
        cmin = int(round(ccenter - 0.5 * width))
        cmax = int(round(ccenter + 0.5 * width))

        return rmin, rmax, cmin, cmax

