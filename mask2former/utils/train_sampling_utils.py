import torch
import torchvision
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from ..data.scribble.gen_scribble import get_iterative_scribbles
import numpy as np
import copy
import cv2
import random

def compute_iou(gt_masks, pred_masks, max_objs=15, iou_thres = 0.90):

    intersections = np.sum(np.logical_and(gt_masks, pred_masks), (1,2))
    unions = np.sum(np.logical_or(gt_masks,pred_masks), (1,2))
    ious = intersections/unions
    bool_indices = (ious<iou_thres)
    indices = torch.topk(torch.tensor(ious), len(ious),largest=False).indices
    return indices[bool_indices][:max_objs]

def compute_fn_iou(gt_masks, pred_masks, bg_mask, max_objs=15, iou_thres = 0.90):

    ious = np.zeros(gt_masks.shape[0])
    for i, (gt_mask, pred_mask) in enumerate(zip(gt_masks,pred_masks)):
        _pred_mask = np.logical_xor(pred_mask,bg_mask)
        intersection = _pred_mask.sum()
        union = gt_mask.sum()
        ious[i] = intersection/union
    bool_indices = (ious<iou_thres)
    indices = torch.topk(torch.tensor(ious), len(ious),largest=False).indices
    return indices[bool_indices][:max_objs]

def _get_next_coords_bg(all_fp, timestamp, max_num_points=2, use_largest_cc = True):

    _probs = [0.80,0.20]
    if use_largest_cc:
        _, labels_im = cv2.connectedComponents(all_fp.astype(np.uint8))
        error_mask = labels_im == np.argmax(np.bincount(labels_im.flat)[1:]) + 1
        error_mask = error_mask.astype(np.uint8)
    else:
        error_mask = all_fp
    fp_mask = np.pad(all_fp, ((1, 1), (1, 1)), 'constant')
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]

    _max_dist = np.max(fp_mask_dt)
    inner_mask = fp_mask_dt > (_max_dist / 2.0)

    sample_locations = np.argwhere(inner_mask)
    if len(sample_locations) > 0:
        _probs = [0.80,0.20]
        num_points = 1+ np.random.choice(np.arange(max_num_points), p=_probs)
        num_points = min(num_points, sample_locations.shape[0])
        
        indices = random.sample(range(sample_locations.shape[0]), num_points)
        points_coords = []
        for index in indices:
            coords = sample_locations[index]
            points_coords.append([coords[0], coords[1],timestamp])
    else:
        return None

def _get_next_coords_fg(pred_mask, gt_mask, timestamp, max_num_points=2):

    fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
    error_mask = np.pad(fn, ((1, 1), (1, 1)), 'constant').astype(np.uint8)
    _probs = [0.80,0.20]

    error_mask_dt = cv2.distanceTransform(error_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
    _max_dist = np.max(error_mask_dt)
    
    inner_mask = error_mask_dt > (_max_dist / 2.0)

    sample_locations = np.argwhere(inner_mask)
    if len(sample_locations) > 0:
        _probs = [0.80,0.20]
        num_points = 1+ np.random.choice(np.arange(max_num_points), p=_probs)
        num_points = min(num_points, sample_locations.shape[0])
        
        indices = random.sample(range(sample_locations.shape[0]), num_points)
        points_coords = []
        for index in indices:
            coords = sample_locations[index]
            points_coords.append([coords[0], coords[1],timestamp])
            # points_coords.append(coords)
        
        return points_coords
    else:
        return None
    
def _get_corrective_clicks(pred_mask, gt_mask, bg_mask, timestamp, max_num_points=2):
    
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5

    # torch functionalities
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)) 
    fp = np.logical_and(fp, bg_mask)
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask)

    is_fg = fn.sum() > fp.sum()
    if is_fg:
        error_mask = np.pad(fn, ((1, 1), (1, 1)), 'constant').astype(np.uint8)
    else:
        error_mask = np.pad(fp, ((1, 1), (1, 1)), 'constant').astype(np.uint8)

    error_mask_dt = cv2.distanceTransform(error_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
    _max_dist = np.max(error_mask_dt)
    
    inner_mask = error_mask_dt > (_max_dist / 2.0)

    sample_locations = np.argwhere(inner_mask)
    if len(sample_locations) > 0:
        _probs = [0.80,0.20]
        num_points = 1+ np.random.choice(np.arange(max_num_points), p=_probs)
        num_points = min(num_points, sample_locations.shape[0])
        
        indices = random.sample(range(sample_locations.shape[0]), num_points)
        points_coords = []
        for index in indices:
            coords = sample_locations[index]
            points_coords.append([coords[0], coords[1],timestamp])
            # points_coords.append(coords)
        
        return (points_coords, is_fg)
    else:
        return None

def get_next_clicks_mq(targets, pred_output, timestamp, device, batched_num_scrbs_per_mask=None,
                                 batched_fg_coords_list=None, batched_bg_coords_list = None, per_obj_sampling = True
):
    
    # OPTIMIZATION
    # directly take targets as input as they are already on the device
    gt_masks_batch= [x['instances'].gt_masks.cpu().numpy() for x in targets]
    pred_masks_batch = [x["instances"].pred_masks.cpu().numpy() for x in pred_output]
    bg_masks_batch = [x['bg_mask'].cpu().numpy() for x in targets]
    
    if per_obj_sampling:
        for i, (gt_masks_per_image, pred_masks_per_image, bg_mask_per_image) in enumerate(zip(gt_masks_batch, pred_masks_batch, bg_masks_batch)):
            
            indices = compute_iou(gt_masks_per_image,pred_masks_per_image)
            
            for j in indices:
                sampled_coords_info = _get_corrective_clicks(pred_masks_per_image[j], gt_masks_per_image[j],
                                                             bg_mask_per_image, timestamp = timestamp, max_num_points=2)
                
                if sampled_coords_info is not None:
                    point_coords, is_fg = sampled_coords_info
                    if is_fg:
                        batched_fg_coords_list[i][j].extend(point_coords)
                        batched_num_scrbs_per_mask[i][j]+= len(point_coords)
                    else:
                        if batched_bg_coords_list[i] is None:
                            batched_bg_coords_list[i] = point_coords
                        else:
                            batched_bg_coords_list[i].extend(point_coords)
    
        return batched_num_scrbs_per_mask, batched_fg_coords_list, batched_bg_coords_list 
    else:
        for i, (gt_masks_per_image, pred_masks_per_image, bg_mask_per_image) in enumerate(zip(gt_masks_batch, pred_masks_batch, bg_masks_batch)):
        
            comb_pred_mask = np.max(pred_masks_per_image,axis=0).astype(np.bool_)
            comb_fg_mask = np.max(gt_masks_per_image,axis=0).astype(np.bool_)
            
            all_fp = np.logical_and(bg_mask_per_image, comb_pred_mask).astype(np.uint8)
            all_fn = np.logical_and(np.logical_not(comb_pred_mask), comb_fg_mask).astype(np.uint8)
            if all_fp.sum() > all_fn.sum():
                sampled_bg_coords = _get_next_coords_bg(all_fp,timestamp)
                if sampled_bg_coords is not None:
                    if batched_bg_coords_list[i] is None:
                        batched_bg_coords_list[i] = sampled_bg_coords
                    else:
                        batched_bg_coords_list[i].extend(sampled_bg_coords)
            else:
                indices = compute_fn_iou(gt_masks_per_image,pred_masks_per_image,bg_mask_per_image)

                for j in indices:
                    sampled_fg_coords = _get_next_coords_fg(pred_masks_per_image[j], gt_masks_per_image[j],
                                                            timestamp = timestamp, max_num_points=2)
                    
                    if sampled_fg_coords is not None:
                        batched_fg_coords_list[i][j].extend(sampled_fg_coords)
                        batched_num_scrbs_per_mask[i][j]+= len(sampled_fg_coords)
        return batched_num_scrbs_per_mask, batched_fg_coords_list, batched_bg_coords_list 
                  