# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import pickle
import numpy as np
import torch
import random
from functools import lru_cache
import cv2
from copy import deepcopy
from detectron2.structures import BoxMode, Boxes
from fvcore.common.timer import Timer
from pycocotools import coco
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances
from detectron2.structures.masks import PolygonMasks

from pycocotools import mask as coco_mask
from dynamite.data.scribble.gen_scribble import get_scribble_eval, get_scribble_gt_mask
from dynamite.data.dataset_mappers.mapper_utils.click_utils import get_clicks_coords
from dynamite.data.dataset_mappers.mapper_utils.datamapper_utils import visualization, filter_instances, build_transform_gen

# from mask2former.data.points.annotation_generator import gen_multi_points_per_mask, generate_point_to_blob_masks

from dynamite.data.points.annotation_generator import create_circular_mask
__all__ = ["COCOLVISSingleInstMQCoordsDatasetMapper"]


def filter_coco_lvis_instances(instances, min_area):
    # num_instances = len(instances.gt_masks)
    # polygon_masks = PolygonMasks(instances.gt_masks.polygons)
    # masks_area = polygon_masks.area()
    m = []
    for mask in instances.gt_masks:
        m.append(mask.sum() > min_area)
    # print(f"instances: {len(instances)}, masks: {len(masks_area)},{masks_area.shape}")
    # num_instances = len(masks_area)
    # m = []
    # for mask_area in masks_area:
        # m.append(mask_area > min_area)
    m = torch.tensor(m).type(torch.bool)
    # print(m)
    return instances[m]    
# This is specifically designed for the COCO dataset.

class COCOLVISSingleInstMQCoordsDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        unique_timestamp,
        random_bg_queries=False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceInteractiveDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.min_area = 1000.0
        self.merge_objects_prob=0.15
        self.random_bg_queries = random_bg_queries
        self.unique_timestamp = unique_timestamp
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "random_bg_queries": cfg.ITERATIVE.TRAIN.RANDOM_BG_QUERIES,
            "unique_timestamp": cfg.ITERATIVE.TRAIN.UNIQUE_TIMESTAMP,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # save_vis(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        stuff_prob = 0
        if "annotations" in dataset_dict:

            # USER: Implement additional transformations if you have other types of data
            if stuff_prob > 0 and random.random() < stuff_prob: 
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if (obj.get("iscrowd", 0) == 0 and obj.get("area",0) > 1000.0) 
                ]
            else:
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if (obj.get("iscrowd", 0) == 0 and obj.get("isThing") and obj.get("area",0) > 1000.0)
                ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape,  mask_format="bitmask")
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not hasattr(instances, 'gt_masks'):
                return None
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # boxes_area = instances.gt_boxes.area()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # print(f"instances before filter:{instances.gt_masks.tensor.shape}")
            # instances = filter_coco_lvis_instances(instances, min_area = 1000.0)
            # visualization(dataset_dict["image"], instances)
            # print(f"instances after filter:{instances.gt_masks.tensor.shape}")
            if len(instances) == 0:
                # print("here")
                return None
            # Generate masks from polygon
            h, w = instances.image_size
           
            if hasattr(instances, 'gt_masks'):
                
                # Make smaller object in front in case of overlapping masks
                
                mask_areas = torch.sum(instances.gt_masks.tensor, (1,2))
                gt_masks = instances.gt_masks.tensor.to(dtype=torch.uint8)
                gt_masks =  gt_masks[sorted(range(len(mask_areas)),key=mask_areas.__getitem__,reverse=True)]

                instance_map = torch.zeros((gt_masks.shape[-2:]), dtype=torch.int16)
                num_objects = gt_masks.shape[0]
                instances_ids = np.arange(1, num_objects + 1)

                for _id, _m in enumerate(gt_masks):
                    instance_map[_m == 1] = _id+1
                    assert (_m != 0).sum() > 0
                
                gt_masks = []
                for _id in instances_ids:
                    _m = (instance_map == _id).to(dtype=torch.uint8)
                    if _m.sum() > 50:
                        gt_masks.append(_m)
                if not len(gt_masks):
                    return None
                gt_masks = torch.stack(gt_masks,dim=0)
                # assert num_objects == gt_masks.shape[0]
                
                # gt_masks = instances.gt_masks.tensor.to(dtype=torch.uint8)
                
                all_masks = dataset_dict["padding_mask"].int()
                # filterd_gt_masks = []
                new_instances = Instances(image_size=image_shape)
                
                # if gt_masks.shape[0] == 1:
                #     num_masks = 1
                # else:
                #     #Take 75% masks as the foreground masks
                #     num_masks = min(int(gt_masks.shape[0]*(0.70)), 30)
                if np.random.rand() < self.merge_objects_prob:
                    num_masks = 2
                else:
                    num_masks = 1
                num_masks =min(gt_masks.shape[0], num_masks)
                random_indices = random.sample(range(gt_masks.shape[0]),num_masks)
                new_gt_masks = gt_masks[random_indices]
                new_gt_masks = torch.max(new_gt_masks,dim=0).values.unsqueeze(0)
                # new_gt_classes = instances.gt_classes[random_indices]

                new_gt_classes = [0]*new_gt_masks.shape[0]
                # new_gt_boxes = instances.gt_masks.get_bounding_boxes()[random_indices]
                new_gt_boxes =  Boxes((np.zeros((new_gt_masks.shape[0],4))))
                
                new_instances.set('gt_masks', new_gt_masks)
                new_instances.set('gt_classes', new_gt_classes)
                new_instances.set('gt_boxes', new_gt_boxes) 
                # filterd_gt_masks = []
                # print(random_indices)
                semantic_map = torch.zeros((new_gt_masks.shape[-2:]), dtype=torch.int16)
                for _id, m in enumerate(new_gt_masks):
                    semantic_map[m == 1] = _id+1
                    all_masks = torch.logical_or(all_masks, m)
                dataset_dict['semantic_map'] = semantic_map
                # new_gt_masks = new_gt_masks.unsqueeze(0)
                # print(new_gt_masks.shape)
                (num_scrbs_per_mask, fg_coords_list, bg_coords_list,
                fg_point_masks, bg_point_masks) = get_clicks_coords(new_gt_masks, all_masks=all_masks, unique_timestamp = self.unique_timestamp)
        
                dataset_dict["fg_scrbs"] = fg_point_masks
                dataset_dict["bg_scrbs"] = bg_point_masks
                dataset_dict["bg_mask"] = torch.logical_not(all_masks).to(dtype = torch.uint8)
                dataset_dict["fg_click_coords"] = fg_coords_list
                dataset_dict["bg_click_coords"] = bg_coords_list
                dataset_dict["num_scrbs_per_mask"] = num_scrbs_per_mask
                # print(masks.tensor.dtype)
                # visualization(dataset_dict["image"], new_instances, prev_output=None, batched_fg_coords_list=[fg_coords_list],batched_bg_coords_list=[bg_coords_list])
                assert len(num_scrbs_per_mask) == new_instances.gt_masks.shape[0]
                assert len(fg_point_masks) == len(num_scrbs_per_mask) 
            else:
                return None

            dataset_dict["instances"] = new_instances

        return dataset_dict