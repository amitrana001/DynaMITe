# Modified by Amit Rana from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
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
from dynamite.data.dataset_mappers.utils import get_clicks_coords, build_transform_gen

__all__ = ["COCOLVISDatasetMapper"]


class COCOLVISDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DynaMITe.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    5. Select atmost 30 random masks from the annotations
    5. Prepare a list of foreground and background clicks for the objects and the background.
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        min_area,
        stuff_prob=0.15,
    ):
        """

        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image
            min_ares: minimum mask area for an object/instance
            stuff_prob: probability to sample stuff category objects
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOLVISDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.min_area = min_area
        self.stuff_prob = stuff_prob
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "stuff_prob": cfg.ITERATIVE.TRAIN.STUFF_PROB,
            "min_area": cfg.INPUT.MIN_AREA_FOR_MASK
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

        if "annotations" in dataset_dict:

            # USER: Implement additional transformations if you have other types of data
            if self.stuff_prob > 0 and random.random() < self.stuff_prob: 
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if (obj.get("iscrowd", 0) == 0 and obj.get("area",0) > self.min_area) 
                ]
            else:
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if (obj.get("iscrowd", 0) == 0 and obj.get("isThing") and obj.get("area",0) > self.min_area)
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

            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            if len(instances) == 0:
                return None
            # Generate masks from polygon
            h, w = instances.image_size
           
            if hasattr(instances, 'gt_masks'):
                
                # Make smaller object in front in case of overlapping masks
                
                # instances.gt_masks.tensor
                gt_masks = instances.gt_masks.tensor.to(dtype=torch.uint8)
                if gt_masks.shape[0] == 1:
                    num_masks = 1
                else:
                    #Take 75% masks as the foreground masks
                    num_masks = min(int(gt_masks.shape[0]*(0.70)), 30)
              
                random_indices = random.sample(range(gt_masks.shape[0]),num_masks)
                gt_masks = gt_masks[random_indices]

                mask_areas = torch.sum(gt_masks, (1,2))
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
                
                all_masks = dataset_dict["padding_mask"].int()
             
                new_instances = Instances(image_size=image_shape)

                new_gt_classes = [0]*gt_masks.shape[0]
                new_gt_boxes =  Boxes((np.zeros((gt_masks.shape[0],4))))
                
                new_instances.set('gt_masks', gt_masks)
                new_instances.set('gt_classes', new_gt_classes)
                new_instances.set('gt_boxes', new_gt_boxes) 
               
                semantic_map = torch.zeros((gt_masks.shape[-2:]), dtype=torch.int16)
                for _id, m in enumerate(gt_masks):
                    semantic_map[m == 1] = _id+1
                    all_masks = torch.logical_or(all_masks, m)
                dataset_dict['semantic_map'] = semantic_map
                
                (num_clicks_per_object, fg_coords_list, bg_coords_list) = get_clicks_coords(gt_masks, all_masks=all_masks)
        
                dataset_dict["bg_mask"] = torch.logical_not(all_masks).to(dtype = torch.uint8)
                dataset_dict["fg_click_coords"] = fg_coords_list
                dataset_dict["bg_click_coords"] = bg_coords_list
                dataset_dict["num_clicks_per_object"] = num_clicks_per_object

                assert len(num_clicks_per_object) == new_instances.gt_masks.shape[0]
            else:
                return None

            dataset_dict["instances"] = new_instances

        return dataset_dict