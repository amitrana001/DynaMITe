
import numpy as np
import torch
import random
from functools import lru_cache
import cv2
import copy
import logging
from copy import deepcopy
from dynamite.data.points.annotation_generator import create_circular_mask
from pycocotools import mask as coco_mask
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances
from detectron2.structures.masks import PolygonMasks

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        # area = coco_mask.area(rles)
        # print(area)
        # if area < min_area:
        #     continue
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def filter_instances(instances, min_area):
    # num_instances = len(instances.gt_masks)
    polygon_masks = PolygonMasks(instances.gt_masks.polygons)
    masks_area = polygon_masks.area()
    # print(f"instances: {len(instances)}, masks: {len(masks_area)},{masks_area.shape}")
    # num_instances = len(masks_area)
    m = []
    for mask_area in masks_area:
        m.append(mask_area > min_area)
    m = torch.tensor(m).type(torch.bool)
    # print(m)
    return instances[m]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if not is_train:
        augmentation = []
        augmentation.append(T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        ))
        return augmentation

    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value = 0)
    ])

    return augmentation

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

def get_clicks_coords(masks, max_num_points=6, first_click_center=True, all_masks=None, t= 0):

    """
    :param masks: numpy array of shape I x H x W
    """
    # assert all_masks is not None
    masks = np.asarray(masks).astype(np.uint8)
    all_masks = np.asarray(all_masks) #.astype(np.uint8)
    _pos_probs = generate_probs(max_num_points, gamma = 0.7)
    _neg_probs = generate_probs(max_num_points+1, gamma = 0.7)

    I, H, W = masks.shape
    num_clicks_per_object = [0]*I
    fg_coords_list = []
    for i, (_m) in enumerate(masks):
        coords = []
  
        if first_click_center:
            center_coords = _point_candidates_dt(_m)
            
            coords.append([center_coords[0], center_coords[1], t])
            t+=1
            num_clicks_per_object[i]+=1 
        
        kernel = np.ones((3,3),np.uint8)
        _eroded_m = cv2.erode(_m,kernel,iterations = 1)
        sample_locations = np.argwhere(_eroded_m)

        num_points =np.random.choice(np.arange(max_num_points), p=_pos_probs)
        num_points = min(num_points,sample_locations.shape[0]//2)

        indices = random.sample(range(sample_locations.shape[0]), num_points)
        for index in indices:
            point_coords = sample_locations[index]

            coords.append([point_coords[0], point_coords[1], t])
            t+=1
            num_clicks_per_object[i]+=1
        fg_coords_list.append(coords)
        
    if np.random.rand() < 0.2:
        return num_clicks_per_object, fg_coords_list, None
    
    bg_coords_list = []
    full_bg_mask = (~all_masks).astype(np.uint8)
    _eroded_bg_mask = cv2.erode(full_bg_mask,kernel,iterations = 1)
    sample_locations = np.argwhere(_eroded_bg_mask)

    num_points =np.random.choice(np.arange(max_num_points+1), p=_neg_probs)
    num_points = min(num_points,sample_locations.shape[0]//2)
    indices = random.sample(range(sample_locations.shape[0]), num_points)

    for index in indices:
        point_coords = sample_locations[index]
       
        bg_coords_list.append([point_coords[0], point_coords[1], t])
        t+=1

    return num_clicks_per_object, fg_coords_list, bg_coords_list