# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances
from detectron2.structures.masks import PolygonMasks

from mask2former.data.dataset_mappers.mapper_utils.datamapper_utils import  build_transform_gen

from mask2former.evaluation.eval_utils import get_gt_clicks_coords_eval

__all__ = ["DAVISSBDMQCoordsEvalMapper"]

# This is specifically designed for the COCO dataset.
class DAVISSBDMQCoordsEvalMapper:
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
        self.min_area = 500.0
        self.unique_timestamp = unique_timestamp
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
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
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
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
            # instances = filter_instances(instances, min_area = 400.0)
            
            if len(instances) == 0:
                # print("here")
                return None
            # Generate masks from polygon
            h, w = instances.image_size
        
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks.tensor

                all_masks = dataset_dict["padding_mask"].int()
                for gt_mask in gt_masks:
                    all_masks = torch.logical_or(all_masks, gt_mask)

                gt_masks = gt_masks.tensor
                (num_scrbs_per_mask, fg_coords_list, bg_coords_list,
                fg_point_masks, bg_point_masks) = get_gt_clicks_coords_eval(gt_masks, unique_timestamp = self.unique_timestamp)
        
                dataset_dict["fg_scrbs"] = fg_point_masks
                dataset_dict["bg_scrbs"] = bg_point_masks
                dataset_dict["bg_mask"] = torch.logical_not(all_masks).to(dtype = torch.uint8)
                dataset_dict["fg_click_coords"] = fg_coords_list
                dataset_dict["bg_click_coords"] = bg_coords_list
                dataset_dict["num_scrbs_per_mask"] = num_scrbs_per_mask
                # print(masks.tensor.dtype)
                # visualization(dataset_dict["image"], new_instances, prev_output=None, batched_fg_coords_list=[fg_coords_list],batched_bg_coords_list=[bg_coords_list])
                assert len(num_scrbs_per_mask) == gt_masks.shape[0]
                assert len(fg_point_masks) == len(num_scrbs_per_mask) 
            else:
                return None

            dataset_dict["instances"] = instances

        return dataset_dict
