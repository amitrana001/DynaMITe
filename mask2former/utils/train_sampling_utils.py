import torch
import torchvision
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from ..data.scribble.gen_scribble import get_iterative_scribbles
import numpy as np
import copy
import cv2
import random
from mask2former.data.points.annotation_generator import create_circular_mask

def compute_iou(gt_masks, pred_masks, max_objs=15, iou_thres = 0.90):

    intersections = np.sum(np.logical_and(gt_masks, pred_masks), (1,2))
    unions = np.sum(np.logical_or(gt_masks,pred_masks), (1,2))
    ious = intersections/unions
    
    indices = torch.topk(torch.tensor(ious), len(ious),largest=False).indices
    worst_indexs = []
    i=0
    while(i<max_objs and i<len(indices)):
        if ious[indices[i]] < iou_thres:
            worst_indexs.append(indices[i])
        i+=1
        if len(worst_indexs)==max_objs:
            break
    return worst_indexs

def compute_fn_iou(gt_masks, pred_masks, bg_mask, max_objs=15, iou_thres = 0.90):

    fn_ratio = np.zeros(gt_masks.shape[0])
    for i, (gt_mask, pred_mask) in enumerate(zip(gt_masks,pred_masks)):
        # _pred_mask = np.logical_and(pred_mask,gt_mask)
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
        fn_area = fn.sum()
        gt_area = gt_mask.sum()
        fn_ratio[i] = fn_area/gt_area
    # bool_indices = (fn_ratio>0)
    # ious = ious[bool_indices]
    indices = torch.topk(torch.tensor(fn_ratio), len(fn_ratio),largest=True).indices
    worst_indexs = []
    i=0
    while(i<max_objs and i < len(indices)):
        if fn_ratio[indices[i]] > 0:
            worst_indexs.append(indices[i])
        i+=1
        if len(worst_indexs)==max_objs:
            break
    return worst_indexs

def _get_next_coords_bg(all_fp, timestamp,device, max_num_points=2, use_largest_cc = True, unique_timestamp=False):

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
        point_masks = []
        H,W = all_fp.shape
        for index in indices:
            coords = sample_locations[index]
            _pm = create_circular_mask(H, W, centers=[coords], radius=3)
            point_masks.append(_pm)
            points_coords.append([coords[0], coords[1],timestamp])
            if unique_timestamp:
                timestamp+=1
        point_masks = torch.from_numpy(np.stack(point_masks, axis=0)).to(device=device, dtype=torch.uint8)
        return (points_coords, point_masks)
    else:
        return None

def _get_next_coords_fg(pred_mask, gt_mask, timestamp, device,max_num_points=2, unique_timestamp=False):

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
        point_masks = []
        H,W = pred_mask.shape
        for index in indices:
            coords = sample_locations[index]
            _pm = create_circular_mask(H, W, centers=[coords], radius=3)
            point_masks.append(_pm)
            points_coords.append([coords[0], coords[1],timestamp])
            if unique_timestamp:
                timestamp+=1
            # points_coords.append(coords)
        point_masks = torch.from_numpy(np.stack(point_masks, axis=0)).to(device=device, dtype=torch.uint8)
        return (points_coords, point_masks)
    else:
        return None
    
def _get_corrective_clicks(pred_mask, gt_mask, bg_mask, timestamp, device, max_num_points=2,unique_timestamp=False):
    
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
        H, W = pred_mask.shape
        points_coords = []
        point_masks = []
        for index in indices:
            coords = sample_locations[index]
            _pm = create_circular_mask(H, W, centers=[coords], radius=3)
            point_masks.append(_pm)
            points_coords.append([coords[0], coords[1],timestamp])
            if unique_timestamp:
                timestamp+=1
            # points_coords.append(coords)
        point_masks = torch.from_numpy(np.stack(point_masks, axis=0)).to(device=device, dtype=torch.uint8)
        return (points_coords, point_masks, is_fg)
    else:
        return None

def get_next_clicks_mq(targets, pred_output, timestamp, device, scribbles = None, batched_num_scrbs_per_mask=None,
                       batched_fg_coords_list=None, batched_bg_coords_list = None, per_obj_sampling = True, 
                       unique_timestamp=False, batched_max_timestamp = None
):
    
    # OPTIMIZATION
    # directly take targets as input as they are already on the device
    gt_masks_batch= [x['instances'].gt_masks.cpu().numpy() for x in targets]
    pred_masks_batch = [x["instances"].pred_masks.cpu().numpy() for x in pred_output]
    bg_masks_batch = [x['bg_mask'].cpu().numpy() for x in targets]
    
    if per_obj_sampling:
        for i, (gt_masks_per_image, pred_masks_per_image, bg_mask_per_image) in enumerate(zip(gt_masks_batch, pred_masks_batch, bg_masks_batch)):
            
            indices = compute_iou(gt_masks_per_image,pred_masks_per_image)
            if unique_timestamp:
                timestamp = batched_max_timestamp[i]+1
            if scribbles:
                for j in indices:
                    sampled_coords_info = _get_corrective_clicks(pred_masks_per_image[j], gt_masks_per_image[j],
                                                                bg_mask_per_image, timestamp = timestamp, device=device,
                                                                max_num_points=2)
                    
                    if sampled_coords_info is not None:
                        point_coords, point_masks, is_fg = sampled_coords_info
                        if unique_timestamp:
                            timestamp += len(point_coords)
                        if is_fg:
                            scribbles[i][j] = torch.cat([scribbles[i][j],point_masks],0)
                            batched_fg_coords_list[i][j].extend(point_coords)
                            batched_num_scrbs_per_mask[i][j]+= len(point_coords)
                        else:
                            if batched_bg_coords_list[i]:
                                batched_bg_coords_list[i].extend(point_coords)
                                scribbles[i][-1] = torch.cat([scribbles[i][-1],point_masks],0)
                            else:
                                batched_bg_coords_list[i] = point_coords
                                scribbles[i][-1] = point_masks
                            assert scribbles[i][-1].shape[0] == len(batched_bg_coords_list[i])
            if unique_timestamp:
                batched_max_timestamp[i] = timestamp-1
        if unique_timestamp:
            return batched_num_scrbs_per_mask, scribbles, batched_fg_coords_list, batched_bg_coords_list, batched_max_timestamp
        return batched_num_scrbs_per_mask, scribbles, batched_fg_coords_list, batched_bg_coords_list
    else:
        if scribbles:
            for i, (gt_masks_per_image, pred_masks_per_image, bg_mask_per_image) in enumerate(zip(gt_masks_batch, pred_masks_batch, bg_masks_batch)):
            
                comb_pred_mask = np.max(pred_masks_per_image,axis=0).astype(np.bool_)
                comb_fg_mask = np.max(gt_masks_per_image,axis=0).astype(np.bool_)
                
                all_fp = np.logical_and(bg_mask_per_image, comb_pred_mask).astype(np.uint8)
                all_fn = np.logical_and(np.logical_not(comb_pred_mask), comb_fg_mask).astype(np.uint8)
                if unique_timestamp:
                    timestamp = batched_max_timestamp[i]+1
                if all_fp.sum() > all_fn.sum():
                    sampled_coords_info = _get_next_coords_bg(all_fp,timestamp,device)
                    if sampled_coords_info is not None:
                        sampled_bg_coords, bg_point_masks = sampled_coords_info

                        if unique_timestamp:
                            timestamp += len(sampled_bg_coords)
                        
                        if batched_bg_coords_list[i]:
                            batched_bg_coords_list[i].extend(sampled_bg_coords)
                            scribbles[i][-1] = torch.cat([scribbles[i][-1],bg_point_masks],0)
                        else:
                            batched_bg_coords_list[i] = sampled_bg_coords
                            scribbles[i][-1] = bg_point_masks
                        assert scribbles[i][-1].shape[0] == len(batched_bg_coords_list[i])
                else:
                    indices = compute_fn_iou(gt_masks_per_image,pred_masks_per_image,bg_mask_per_image)

                    for j in indices:
                        sampled_coords_info = _get_next_coords_fg(pred_masks_per_image[j], gt_masks_per_image[j],
                                                                timestamp = timestamp, device=device, max_num_points=2)
                        
                        if sampled_coords_info is not None:
                            sampled_fg_coords, fg_point_masks = sampled_coords_info
                            
                            if unique_timestamp:
                                timestamp += len(sampled_fg_coords)
                            scribbles[i][j] = torch.cat([scribbles[i][j],fg_point_masks],0)
                            batched_fg_coords_list[i][j].extend(sampled_fg_coords)
                            batched_num_scrbs_per_mask[i][j]+= len(sampled_fg_coords)
                if unique_timestamp:
                    batched_max_timestamp[i] = timestamp-1
            if unique_timestamp:
                return batched_num_scrbs_per_mask, scribbles, batched_fg_coords_list, batched_bg_coords_list, batched_max_timestamp
            return batched_num_scrbs_per_mask, scribbles, batched_fg_coords_list, batched_bg_coords_list  

def get_next_clicks_mq_argmax(targets, pred_output, timestamp, device, scribbles = None, batched_num_scrbs_per_mask=None,
                       batched_fg_coords_list=None, batched_bg_coords_list = None, per_obj_sampling = True, 
                       unique_timestamp=False, batched_max_timestamp = None
):
    
    # OPTIMIZATION
    # directly take targets as input as they are already on the device
    gt_masks_batch= [x['instances'].gt_masks.cpu().numpy() for x in targets]
    pred_masks_batch = [x["instances"].pred_masks.cpu().numpy() for x in pred_output]
    padding_masks_batch = [x['padding_mask'].cpu().numpy() for x in targets]
    semantic_maps_batch = [x['semantic_map'].cpu().numpy() for x in targets]
    
    
    for i, (gt_masks_per_image, pred_masks_per_image, semantic_map, padding_mask) in enumerate(zip(gt_masks_batch, pred_masks_batch, semantic_maps_batch,padding_masks_batch)):
        
        indices = compute_iou(gt_masks_per_image,pred_masks_per_image)
        if unique_timestamp:
            timestamp = batched_max_timestamp[i]+1
        if scribbles:
            for j in indices:
                sampled_coords_info = _get_corrective_clicks_argmax(pred_masks_per_image[j], gt_masks_per_image[j],
                                                            semantic_map, padding_mask, timestamp = timestamp,
                                                            unique_timestamp=unique_timestamp,
                                                            device=device, radius=3, max_num_points=2)
                
                if sampled_coords_info is not None:
                    point_coords, point_masks, obj_indices = sampled_coords_info
                    if unique_timestamp:
                        timestamp += len(point_coords)
                    for k, obj_indx in enumerate(obj_indices):
                        if obj_indx == -1:
                            if batched_bg_coords_list[i]:
                                batched_bg_coords_list[i].extend([point_coords[k]])
                                scribbles[i][-1] = torch.cat([scribbles[i][-1],point_masks[k].unsqueeze(0)],0)
                            else:
                                batched_bg_coords_list[i] = [point_coords[k]]
                                scribbles[i][-1] = point_masks[k].unsqueeze(0)
                            assert scribbles[i][-1].shape[0] == len(batched_bg_coords_list[i])
                        else:
                            scribbles[i][obj_indx] = torch.cat([scribbles[i][obj_indx],point_masks[k].unsqueeze(0)],0)
                            batched_fg_coords_list[i][obj_indx].extend([point_coords[k]])
                            batched_num_scrbs_per_mask[i][obj_indx]+= 1
        if unique_timestamp:
            batched_max_timestamp[i] = timestamp-1
    if unique_timestamp:
        return batched_num_scrbs_per_mask, scribbles, batched_fg_coords_list, batched_bg_coords_list, batched_max_timestamp
    return batched_num_scrbs_per_mask, scribbles, batched_fg_coords_list, batched_bg_coords_list
                  
# from mask2former.data.points.annotation_generator import create_circular_mask, get_max_dt_point_mask
def _get_corrective_clicks_argmax(pred_mask, gt_mask, semantic_map, padding_mask,timestamp,
                   unique_timestamp, device, radius=3,  max_num_points=2
):
    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    padding_mask = np.asarray(padding_mask, dtype = np.bool_)

    fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)
    
    fn_mask = np.logical_and(fn_mask, np.logical_not(padding_mask))
    fp_mask = np.logical_and(fp_mask, np.logical_not(padding_mask))
   
    H, W = gt_mask.shape

    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
    fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
    fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    if fn_max_dist > fp_max_dist:
        inner_mask = fn_mask_dt > (fn_max_dist / 2.0)
    else:
        inner_mask = fp_mask_dt > (fp_max_dist / 2.0)

    sample_locations = np.argwhere(inner_mask)
    if len(sample_locations) > 0:
        _probs = [0.80,0.20]
        num_points = 1+ np.random.choice(np.arange(max_num_points), p=_probs)
        num_points = min(num_points, sample_locations.shape[0])
        
        indices = random.sample(range(sample_locations.shape[0]), num_points)
        H, W = pred_mask.shape
        points_coords = []
        point_masks = []
        obj_indices = []
        for index in indices:
            coords = sample_locations[index]
            _pm = create_circular_mask(H, W, centers=[coords], radius=3)
            point_masks.append(_pm)
            points_coords.append([coords[0], coords[1],timestamp])
            obj_indx = semantic_map[coords[0]][coords[1]] -1
            obj_indices.append(obj_indx)
            if unique_timestamp:
                timestamp+=1
            # points_coords.append(coords)
        point_masks = torch.from_numpy(np.stack(point_masks, axis=0)).to(device=device, dtype=torch.uint8)
        return (points_coords, point_masks, obj_indices)
    else:
        None