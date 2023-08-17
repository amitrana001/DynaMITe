import math
import torch
import numpy as np
import copy
import cv2
import random

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

def get_next_clicks(targets, pred_output, timestamp, batched_num_clicks_per_object=None,
                       fg_coords=None, bg_coords = None,
                       max_timestamp = None
):
    
    # OPTIMIZATION
    # directly take targets as input as they are already on the device
    gt_masks_batch= [x['instances'].gt_masks.cpu().numpy() for x in targets]
    pred_masks_batch = [x["instances"].pred_masks.cpu().numpy() for x in pred_output]
    padding_masks_batch = [x['padding_mask'].cpu().numpy() for x in targets]
    semantic_maps_batch = [x['semantic_map'].cpu().numpy() for x in targets]
    
    
    for i, (gt_masks_per_image, pred_masks_per_image, semantic_map, padding_mask) in enumerate(zip(gt_masks_batch, pred_masks_batch, semantic_maps_batch,padding_masks_batch)):
        
        indices = compute_iou(gt_masks_per_image,pred_masks_per_image)
        # if unique_timestamp:
        timestamp = max_timestamp[i]+1
        # if scribbles:
        for j in indices:
            sampled_coords_info = _get_corrective_clicks(pred_masks_per_image[j], gt_masks_per_image[j],
                                                        semantic_map, padding_mask, timestamp = timestamp,
                                                        max_num_points=2)
            
            if sampled_coords_info is not None:
                point_coords, obj_indices = sampled_coords_info
                # if unique_timestamp:
                timestamp += len(point_coords)
                for k, obj_indx in enumerate(obj_indices):
                    if obj_indx == -1:
                        if bg_coords[i]:
                            bg_coords[i].extend([point_coords[k]])
                        else:
                            bg_coords[i] = [point_coords[k]]
                    else:
                        fg_coords[i][obj_indx].extend([point_coords[k]])
                        batched_num_clicks_per_object[i][obj_indx]+= 1
        # if unique_timestamp:
        max_timestamp[i] = timestamp-1

        return batched_num_clicks_per_object,  fg_coords, bg_coords, max_timestamp
                  
def _get_corrective_clicks(pred_mask, gt_mask, semantic_map, padding_mask,
                           timestamp, max_num_points=2,
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
        obj_indices = []
        for index in indices:
            coords = sample_locations[index]
        
            points_coords.append([coords[0], coords[1],timestamp])
            obj_indx = semantic_map[coords[0]][coords[1]] -1
            obj_indices.append(obj_indx)
            timestamp+=1
        return (points_coords, obj_indices)
    else:
        None

def get_spatiotemporal_embeddings(pos_tensor, positional_embeddings):
        
        scale = 2 * math.pi
        if positional_embeddings == "temporal":
            dim_t = torch.arange(256, dtype=torch.float, device=pos_tensor.device)
            dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / 256)
            t_embed = pos_tensor[:, :, 2] * scale
            pos_t = t_embed[:, :, None] / dim_t
            pos_t[:, :, 0::2][torch.where(pos_t[:, :, 0::2] < 0)] = 0.0
            pos_t[:, :, 1::2][torch.where(pos_t[:, :, 1::2] < 0)] = math.pi/2
            pos_t = torch.stack((pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3).flatten(2)
            return pos_t
        dim_t = torch.arange(128, dtype=torch.float, device=pos_tensor.device)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / 128)
        x_embed = pos_tensor[:, :, 1] * scale
        y_embed = pos_tensor[:, :, 0] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x[:, :, 0::2][torch.where(pos_x[:, :, 0::2] < 0)] = 0.0
        pos_x[:, :, 1::2][torch.where(pos_x[:, :, 1::2] < 0)] = math.pi/2
        pos_y[:, :, 0::2][torch.where(pos_y[:, :, 0::2] < 0)] = 0.0
        pos_y[:, :, 1::2][torch.where(pos_y[:, :, 1::2] < 0)] = math.pi/2
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        if positional_embeddings == "spatial":
            return torch.cat((pos_y, pos_x), dim=2)
        elif positional_embeddings == "spatio_temporal":
            t_embed = pos_tensor[:, :, 2] * scale
            pos_t = t_embed[:, :, None] / dim_t
            pos_t[:, :, 0::2][torch.where(pos_t[:, :, 0::2] < 0)] = 0.0
            pos_t[:, :, 1::2][torch.where(pos_t[:, :, 1::2] < 0)] = math.pi/2
            pos_t = torch.stack((pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3).flatten(2)
            
            pos = torch.cat((pos_y, pos_x, pos_t), dim=2)
            return pos
    
def get_pos_tensor_coords(fg_coords, bg_coords, num_queries, height, width, device, max_timestamp=None):

    #fg_coords: batch x (list of list of fg coords) [y,x,t]

    # return
    # points: Bs x num_queries x 3 
    B = len(fg_coords)
    
    pos_tensor = []
    
    for i, fg_coords_per_image in enumerate(fg_coords):
        coords_per_image  = []
        if max_timestamp is not None:
            t = max(max_timestamp[i],500)
        for fg_coords_per_mask in fg_coords_per_image:
            for coords in fg_coords_per_mask:
                if max_timestamp is not None:
                    coords_per_image.append([coords[0]/height, coords[1]/width, coords[2]/t])
                else:
                    coords_per_image.append([coords[0]/height, coords[1]/width, coords[2]])
        if bg_coords[i] is not None:
            for coords in bg_coords[i]:
                if max_timestamp is not None:
                    coords_per_image.append([coords[0]/height, coords[1]/width, coords[2]/t])
                else:
                    coords_per_image.append([coords[0]/height, coords[1]/width, coords[2]])
        coords_per_image.extend([[-1.0,-1.0,-1.0]] * (num_queries-len(coords_per_image)))
        pos_tensor.append(torch.tensor(coords_per_image,device=device))
    pos_tensor = torch.stack(pos_tensor)
    return pos_tensor