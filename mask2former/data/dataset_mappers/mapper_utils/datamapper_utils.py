import copy
import logging

import numpy as np
import torch
import random
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
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation

import cv2
from detectron2.utils.visualizer import Visualizer

def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))
color_map = get_palette(80)[1:]

def visualization(image, instances, prev_output=None, batched_fg_coords_list=None,batched_bg_coords_list=None,
                  alpha_blend=0.6, num_iter = 0):
    import copy
    image = copy.deepcopy(image.cpu())
    batched_fg_coords_list = copy.deepcopy(batched_fg_coords_list)
    batched_bg_coords_list = copy.deepcopy(batched_bg_coords_list)
    image = np.asarray(image.permute(1,2,0))
    visualizer = Visualizer(image, metadata=None)
    if prev_output is not None:
        import torchvision.transforms.functional as F
        pred_masks = F.resize(prev_output.pred_masks.detach().to(dtype=torch.uint8), image.shape[:2])
    else:
        pred_masks = instances.gt_masks.cpu()
    c = []
    for i in range(pred_masks.shape[0]):
        # c.append(color_map[2*(i)+2]/255.0)
        c.append(color_map[i]/255.0)
    # pred_masks = np.asarray(pred_masks).astype(np.bool_)
    vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c, alpha=alpha_blend)
    # [Optional] prepare labels

    image = vis.get_image()
    # # Laminate your image!
    total_colors = len(color_map)-1
    
    h,w = image.shape[:2]
    if batched_fg_coords_list is not None:
        
        for j, fg_coords_per_mask in enumerate(batched_fg_coords_list[0]):
            for i, coords in enumerate(fg_coords_per_mask):
                color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                if i==0:
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), 8, tuple(color), -1)
                else:
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), 3, tuple(color), -1)
        
        if batched_bg_coords_list[0]:
            for i, coords in enumerate(batched_bg_coords_list[0]):
                color = np.array([255,0,0], dtype=np.uint8)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                image = cv2.circle(image, (int(coords[1]), int(coords[0])), 3, tuple(color), -1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # image = cv2.resize(image, (inputs["width"],inputs["height"]))
    # save_dir = os.path.join("./train_vis/", str(batched_inputs[0]['image_id']))
    # os.makedirs(save_dir, exist_ok=True)
    # cv2.imwrite(os.path.join(save_dir, f"iter_{num_iter}.jpg"), image)

def save_vis(dataset_dict, prev_output=None, batched_fg_coords_list=None,batched_bg_coords_list=None,
                  alpha_blend=0.6, num_iter = 0):
    import copy
    dataset_dict = copy.deepcopy(dataset_dict)
    # image = copy.deepcopy(image.cpu())
    # image = np.asarray(image.permute(1,2,0))
    image = utils.read_image(dataset_dict["file_name"], format="RGB")
    utils.check_image_size(dataset_dict, image)
    # image = image.cpu()
    image_shape = image.shape[:2]  # h, w
    stuff_prob = 0
    if "annotations" in dataset_dict:

        # USER: Implement additional transformations if you have other types of data
        if stuff_prob > 0 and random.random() < stuff_prob: 
            annos = [
                utils.transform_instance_annotations(obj, None, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            annos = [
                utils.transform_instance_annotations(obj, None, image_shape)
                for obj in dataset_dict.pop("annotations")
                if (obj.get("iscrowd", 0) == 0 and obj.get("isThing"))
            ]
    # NOTE: does not support BitMask due to augmentation
    # Current BitMask cannot handle empty objects
    instances = utils.annotations_to_instances(annos, image_shape,  mask_format="bitmask")
    if not hasattr(instances, 'gt_masks'):
        return None
    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
    # boxes_area = instances.gt_boxes.area()
    # Need to filter empty instances first (due to augmentation)
    instances = utils.filter_empty_instances(instances)
    # print(f"instances before filter:{instances.gt_masks.tensor.shape}")
    # instances = filter_coco_lvis_instances(instances, min_area = 1000.0)

    visualizer = Visualizer(image, metadata=None)
    if prev_output is not None:
        import torchvision.transforms.functional as F
        pred_masks = F.resize(prev_output.pred_masks.detach().to(dtype=torch.uint8), image.shape[:2])
    else:
        pred_masks = instances.gt_masks.tensor.cpu()
    c = []
    for i in range(pred_masks.shape[0]):
        # c.append(color_map[2*(i)+2]/255.0)
        c.append(color_map[i]/255.0)
    # pred_masks = np.asarray(pred_masks).astype(np.bool_)
    vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c, alpha=alpha_blend)
    # [Optional] prepare labels

    image = vis.get_image()
    # # Laminate your image!
    total_colors = len(color_map)-1
    
    h,w = image.shape[:2]
    if batched_fg_coords_list is not None:
        for j, fg_coords_per_mask in enumerate(batched_fg_coords_list[0]):
            for i, coords in enumerate(fg_coords_per_mask):
                color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                if i==0:
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), 8, tuple(color), -1)
                else:
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), 3, tuple(color), -1)
        
        if batched_bg_coords_list[0]:
            for i, coords in enumerate(batched_bg_coords_list[0]):
                color = np.array([255,0,0], dtype=np.uint8)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                image = cv2.circle(image, (int(coords[1]), int(coords[0])), 3, tuple(color), -1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Image",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    import os
    # image = cv2.resize(image, (inputs["width"],inputs["height"]))
    save_dir = os.path.join("./train_vis/datamapper/", str(dataset_dict['image_id']))
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"orig_res.jpg"), image)