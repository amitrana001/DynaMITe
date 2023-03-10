
import numpy as np
import torch
import random
from functools import lru_cache
import cv2
from copy import deepcopy
from mask2former.data.points.annotation_generator import create_circular_mask

def _point_candidates_dt(mask, k=1.7):
    mask = mask.astype(np.uint8)

    padded_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    candidates = np.argwhere(dt > (dt.max()/k))
    indices = np.random.randint(0,candidates.shape[0])
    return candidates[indices]

@lru_cache(maxsize=None)
def generate_probs(max_num_points, gamma):
    probs = []
    last_value = 1
    for i in range(max_num_points):
        probs.append(last_value)
        last_value *= gamma

    probs = np.array(probs)
    probs /= probs.sum()

    return probs

def get_clicks_coords(masks, max_num_points=6, radius_size=8, first_click_center=True, all_masks=None, t= 0, unique_timestamp=False,
    use_point_features = True):

    """
    :param masks: numpy array of shape I x H x W
    :param patch_size: size of patch (int)
    """
    # assert all_masks is not None
    masks = np.asarray(masks).astype(np.uint8)
    all_masks = np.asarray(all_masks) #.astype(np.uint8)
    _pos_probs = generate_probs(max_num_points, gamma = 0.7)
    _neg_probs = generate_probs(max_num_points+1, gamma = 0.7)

    I, H, W = masks.shape
    num_scrbs_per_mask = [0]*I
    fg_coords_list = []
    fg_point_masks = []
    for i, (_m) in enumerate(masks):
        coords = []
        point_masks_per_obj = []
        if first_click_center:
            center_coords = _point_candidates_dt(_m)
            # center_coords.append(t)
            if not use_point_features:
                _pm = create_circular_mask(H, W, centers=[center_coords], radius=radius_size)
                point_masks_per_obj.append(_pm)
            coords.append([center_coords[0], center_coords[1], t])
            if unique_timestamp:
                t+=1
            num_scrbs_per_mask[i]+=1 
        
        kernel = np.ones((3,3),np.uint8)
        _eroded_m = cv2.erode(_m,kernel,iterations = 1)
        sample_locations = np.argwhere(_eroded_m)

        num_points =np.random.choice(np.arange(max_num_points), p=_pos_probs)
        num_points = min(num_points,sample_locations.shape[0]//2)

        indices = random.sample(range(sample_locations.shape[0]), num_points)
        for index in indices:
            point_coords = sample_locations[index]
            if not use_point_features:
                _pm = create_circular_mask(H, W, centers=[point_coords], radius=3)
                point_masks_per_obj.append(_pm)

            coords.append([point_coords[0], point_coords[1], t])
            if unique_timestamp:
                t+=1
            num_scrbs_per_mask[i]+=1
        fg_coords_list.append(coords)
        if not use_point_features:
            fg_point_masks.append(torch.from_numpy(np.stack(point_masks_per_obj, axis=0)).to(torch.uint8))
    
    if np.random.rand() < 0.2:
        return num_scrbs_per_mask, fg_coords_list, None, fg_point_masks, None
    
    bg_coords_list = []
    bg_point_masks = None
    full_bg_mask = (~all_masks).astype(np.uint8)
    _eroded_bg_mask = cv2.erode(full_bg_mask,kernel,iterations = 1)
    sample_locations = np.argwhere(_eroded_bg_mask)

    num_points =np.random.choice(np.arange(max_num_points+1), p=_neg_probs)
    num_points = min(num_points,sample_locations.shape[0]//2)
    indices = random.sample(range(sample_locations.shape[0]), num_points)
    point_masks_per_bg = []
    for index in indices:
        point_coords = sample_locations[index]
        if not use_point_features:
            _pm = create_circular_mask(H, W, centers=[point_coords], radius=3)
            point_masks_per_bg.append(_pm)
        bg_coords_list.append([point_coords[0], point_coords[1], t])
        if unique_timestamp:
            t+=1
    if len(point_masks_per_bg):
        bg_point_masks = torch.from_numpy(np.stack(point_masks_per_bg, axis=0)).to(torch.uint8)    

    return num_scrbs_per_mask, fg_coords_list, bg_coords_list, fg_point_masks, bg_point_masks