# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch
import random
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances
from detectron2.structures.masks import PolygonMasks

from pycocotools import mask as coco_mask
from mask2former.data.scribble.gen_scribble import get_scribble_gt, get_scribble_gt_mask
# from coco_instance_interactive_dataset_mapper import filter_instances, build_transform_gen, convert_coco_poly_to_mask
from mask2former.data.points.annotation_generator import generate_point_to_blob_masks
from mask2former.data.dataset_mappers.mapper_utils.datamapper_utils import convert_coco_poly_to_mask, filter_instances, build_transform_gen

__all__ = ["COCOAllInstClicksDatasetMapper"]

# This is specifically designed for the COCO dataset.
class COCOAllInstClicksDatasetMapper:
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
        self.min_area = 500.0
        self.random_bg_queries = random_bg_queries
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "random_bg_queries": cfg.ITERATIVE.TRAIN.RANDOM_BG_QUERIES
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        distractor_objects= False
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

        # if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            # return dataset_dict

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
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            # if not hasattr(instances, 'gt_masks'):
            #     # print("no attribue gt_masks")
            #     return None
            # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # boxes_area = instances.gt_boxes.area()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)

            # print(f"instances_before_filter:{len(instances)}")
            if self.is_train:
                instances = filter_instances(instances, min_area = 400.0)
            
            if len(instances) == 0:
                # print("zero instances after filter")
                return None
            # Generate masks from polygon
            # print(f"instances_after_filter:{len(instances)}")
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            # no_gt_masks = False
            # polygon_masks = PolygonMasks(instances.gt_masks.polygons)
            # gt_masks_area = polygon_masks.area()
            # dataset_dict['polygons'] = instances.gt_masks
            if hasattr(instances, 'gt_masks'):
                
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                all_masks = dataset_dict["padding_mask"].int()
                
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances.gt_masks = gt_masks
                for m in gt_masks:
                    all_masks = torch.logical_or(all_masks, m)
                gt_masks = gt_masks.unsqueeze(0)
                fg_masks, bg_masks = generate_point_to_blob_masks(gt_masks, all_masks=all_masks)
                dataset_dict["fg_scrbs"] = fg_masks.squeeze(0)
                # only_bg_mask =  get_scribble_gt_mask(np.asarray(all_masks).astype(np.uint8)*255, bg = True)
                if not distractor_objects:
                    dataset_dict["bg_scrbs"] = bg_masks
                # else:
                #     dataset_dict["bg_scrbs"] = torch.cat((only_bg_mask, torch.stack(bg_scrbs,0)),0)
                dataset_dict["scrbs_count"] = dataset_dict["fg_scrbs"].shape[0] + dataset_dict["bg_scrbs"].shape[0]
            else:
                return None
            if self.is_train:
                dataset_dict["instances"] = instances
            else:
                # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                dataset_dict["instances"] = instances

        return dataset_dict
