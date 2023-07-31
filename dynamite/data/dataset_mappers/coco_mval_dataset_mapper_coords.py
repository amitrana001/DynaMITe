# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import os
import numpy as np
import torch
import torchvision
import random
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances
from detectron2.structures.masks import PolygonMasks
import pickle
from torchvision import transforms
from pycocotools import mask as coco_mask
from dynamite.data.scribble.gen_scribble import get_scribble_gt, get_scribble_gt_mask
# from coco_instance_interactive_dataset_mapper import filter_instances, build_transform_gen, convert_coco_poly_to_mask
from dynamite.data.points.annotation_generator import get_gt_points_determinstic, generate_point_to_blob_masks_eval_deterministic
from dynamite.data.scribble.gen_scribble import get_scribble_eval, get_scribble_gt_mask
from .mapper_utils.datamapper_utils import build_transform_gen, convert_coco_poly_to_mask
from dynamite.evaluation.eval_utils import get_gt_clicks_coords_eval
import torch.nn.functional as F
__all__ = ["COCOMvalCoordsDatasetMapper"]

class COCOMvalCoordsDatasetMapper:
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
        is_train=False,
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
            "[COCOMvalCoordsDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.min_area = 500.0
        self.unique_timestamp = unique_timestamp
        
    @classmethod
    def from_config(cls, cfg, is_train=False):
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

        from detectron2.data import transforms as T
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

        if "instances" in dataset_dict:
            
            instances = dataset_dict['instances']
            if len(instances) == 0:
                # print("zero instances after filter")
                return None
            # Generate masks from polygon
            # print(f"instances_after_filter:{len(instances)}")
            h, w = instances.image_size
            
            if hasattr(instances, 'gt_masks'):
                trans = torchvision.transforms.Resize(image_shape)
                gt_masks = instances.gt_masks
                gt_masks = trans(gt_masks)
                # gt_masks = F.interpolate(gt_masks.unsqueeze(0), (image_shape[0], image_shape[1]), mode='bilinear', align_corners=False)
                # gt_masks = gt_masks.squeeze(0)
                new_instances = Instances(image_size=image_shape)
                all_masks = dataset_dict["padding_mask"].int()
                
                new_instances.set('gt_masks', gt_masks)
                new_instances.set('gt_classes', instances.gt_classes)
                new_instances.set('gt_boxes', instances.gt_boxes) 
                   
                # gt_masks = gt_masks
                ignore_masks = None
                if 'ignore_mask' in dataset_dict:
                    ignore_masks = dataset_dict['ignore_mask'].to(device='cpu', dtype = torch.uint8)
                    ignore_masks =  trans(ignore_masks)
                    # ignore_masks = F.interpolate(ignore_masks.unsqueeze(0), (image_shape[0], image_shape[1]), mode='bilinear', align_corners=False)
                    # ignore_masks = ignore_masks.squeeze(0)
                # fg_scrbs, num_scrbs_per_mask, coords = get_gt_points_determinstic(gt_masks, max_num_points=1, ignore_masks=ignore_masks)
                
                (num_scrbs_per_mask, fg_coords_list, bg_coords_list,
                fg_point_masks, bg_point_masks) = get_gt_clicks_coords_eval(gt_masks, ignore_masks=ignore_masks, unique_timestamp=self.unique_timestamp)
        
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
