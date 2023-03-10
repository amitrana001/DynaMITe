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
import copy

class zoomIn:

    def __init__(self, cfg, gt_masks, inputs, model, expansion_ratio=1.4):
        self.expansion_ratio = expansion_ratio
        self.gt_masks = gt_masks #1 x H_t x W_t
        self.H_t, self.W_t = gt_masks.shape[-2:]
        self.inputs = inputs
        self.object_roi = None
        self.model = model
        #test_time transforms 
        self.augmentation = []
        self.augmentation.append(T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        ))

    def apply_zoom(self, click_coords, inputs, pred_masks, images, scribbles, 
                    num_insts, features, mask_features, transformer_encoder_features,
                    multi_scale_features, prev_mask_logits,
                    batched_num_scrbs_per_mask,
                    batched_fg_coords_list, batched_bg_coords_list,
                    batched_max_timestamp
    ):

        # pred_masks: H_t x W_t
        mask_with_zoom = copy.deepcopy(pred_masks)
        # scribbles = sem_seg_postprocess(scribbles[0].to('cpu'), images.image_sizes[0], self.H_t, self.W_t)
        mask_features = copy.deepcopy(mask_features)
        multi_scale_features = copy.deepcopy(multi_scale_features)
        self.object_roi = get_object_roi(pred_masks[0], click_coords, self.expansion_ratio)
        rmin, rmax, cmin, cmax = self.object_roi
       
        h_padded,w_padded = images.tensor.shape[-2:]
        h_m,w_m = mask_features.shape[-2:]
        mask_features_resized = torchvision.transforms.Resize(size = (h_padded,w_padded))(mask_features)
        mask_features_resized = mask_features_resized[:,:,rmin:rmax + 1, cmin:cmax + 1]
        mask_features_resized =  torchvision.transforms.Resize(size = (h_m,w_m))(mask_features_resized)

        multi_scale_features_resized = []
        for i in range(3):
            
            feat = multi_scale_features[i]
            h_f, w_f = feat.shape[-2:]
            feat = torchvision.transforms.Resize(size = (h_padded,w_padded))(feat)
            feat = feat[:,:,rmin:rmax + 1, cmin:cmax + 1]
            feat =  torchvision.transforms.Resize(size = (h_f,w_f))(feat)
            multi_scale_features_resized.append(feat)

        new_batched_num_scrbs_per_mask = [[0]]
        new_batched_fg_coords_list = []
        new_obj_coords = []
        obj_coords = batched_fg_coords_list[0][0]
        for coords in obj_coords:
            if (coords[0]>=rmin and coords[0]<= rmax) and(coords[1]>=cmin and coords[1]<=cmax):
                new_obj_coords.append(coords)
                new_batched_num_scrbs_per_mask[0][0]+=1
        new_batched_fg_coords_list.append([new_obj_coords])
        
        new_batched_bg_coords_list =[]
        if batched_bg_coords_list[0] is None:
            new_batched_bg_coords_list= [None]
        else:
            new_obj_coords = []
            for coords in batched_bg_coords_list[0]:
                if (coords[0]>=rmin and coords[0]<= rmax) and(coords[1]>=cmin and coords[1]<=cmax):
                    new_obj_coords.append(coords)
            new_batched_bg_coords_list.append(new_obj_coords) 
        

        processed_results = self.model(inputs, images, scribbles, num_insts,
                                        features, mask_features_resized, transformer_encoder_features,
                                        multi_scale_features_resized, prev_mask_logits,
                                        new_batched_num_scrbs_per_mask,
                                        new_batched_fg_coords_list, new_batched_bg_coords_list,
                                        batched_max_timestamp = batched_max_timestamp)[0]
        
        pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
        pred_masks = torchvision.transforms.Resize(size = (rmax-rmin+1,cmax-cmin+1))(pred_masks)

        # roi_input = copy.deepcopy(inputs)
        # cropped_image = np.array(inputs[0]['image']).transpose(1,2,0)[rmin:rmax + 1, cmin:cmax + 1,:]
        # crop_height, crop_width = cropped_image.shape[:2]

        # image, transforms = T.apply_transform_gens(self.augmentation, cropped_image)
        # roi_input[0]['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # roi_input[0]['height'] = crop_height
        # roi_input[0]['width']  = crop_width

        # image_shape = image.shape[:2]
        # trans = torchvision.transforms.Resize(image_shape)
        # roi_input[0]['fg_scrbs'] = trans(inputs[0]['fg_scrbs'][:,rmin:rmax + 1, cmin:cmax + 1])
        # roi_input[0]['bg_scrbs'] = None
        # if inputs[0]['bg_scrbs'] is not None:
        #     roi_input[0]['bg_scrbs'] = trans(inputs[0]['bg_scrbs'][:,rmin:rmax + 1, cmin:cmax + 1])

        # processed_results, _, _, _, _, _, _, _, _ = self.model(roi_input)
        # roi_mask = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
        # roi_mask = torchvision.transforms.Resize(size = (crop_height,crop_width))(roi_mask)
        mask_with_zoom[:,rmin:rmax + 1, cmin:cmax + 1] = pred_masks[0]
        return mask_with_zoom, [rmin,rmax,cmin,cmax] 

def get_object_roi(pred_mask, click_coords, expansion_ratio, min_crop_size=None):
    # pred_mask = pred_mask.copy()

    # for click in clicks_list:
    #     if click.is_positive:
    #         pred_mask[int(click.coords[0]), int(click.coords[1])] = 1
    # all_scribbles = torch.logical_or(scribbles,0)
    # all_scribbles = torch.max(scribbles,0).values
    # total_mask = torch.logical_or(pred_mask, all_scribbles)

    pred_mask[click_coords[0]][click_coords[1]] = 1
    total_mask = np.asarray(pred_mask)
    bbox = get_bbox_from_mask(total_mask)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    return bbox

def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expansion_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = np.asarray(bbox)
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expansion_ratio * (rmax - rmin + 1)
    width = expansion_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax

def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (max(rmin, bbox[0]), min(rmax, bbox[1]),
            max(cmin, bbox[2]), min(cmax, bbox[3]))


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union