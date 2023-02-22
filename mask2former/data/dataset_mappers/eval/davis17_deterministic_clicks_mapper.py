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

from pycocotools import mask as coco_mask
from mask2former.data.scribble.gen_scribble import get_scribble_eval, get_scribble_gt_mask
from mask2former.data.dataset_mappers.mapper_utils.datamapper_utils import convert_coco_poly_to_mask, filter_instances, build_transform_gen

from mask2former.data.points.annotation_generator import get_gt_points_determinstic, generate_point_to_blob_masks_eval_deterministic

__all__ = ["DAVIS17DetmClicksDatasetMapper"]

# This is specifically designed for the COCO dataset.
class DAVIS17DetmClicksDatasetMapper:
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
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
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
            # instances = filter_instances(instances, min_area = 400.0)
            
            if len(instances) == 0:
                # print("here")
                return None
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            # no_gt_masks = False
            # polygon_masks = PolygonMasks(instances.gt_masks.polygons)
            # gt_masks_area = polygon_masks.area()
            # dataset_dict['polygons'] = instances.gt_masks
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks.tensor

                fg_scribs = []
                all_masks = dataset_dict["padding_mask"].int()
                # filterd_gt_masks = []
                
                for gt_mask in gt_masks:
                    all_masks = torch.logical_or(all_masks, gt_mask)

                gt_masks = gt_masks.tensor
                fg_scrbs, num_scrbs_per_mask, coords = get_gt_points_determinstic(gt_masks, max_num_points=1)
                
                dataset_dict["fg_scrbs"] = fg_scrbs
                dataset_dict["num_scrbs_per_mask"] = num_scrbs_per_mask
                dataset_dict["bg_scrbs"] = None
                # dataset_dict["bg_scrbs"] = bg_scrbs
                # dataset_dict["bg_scrbs"] = torch.zeros_like(bg_scrbs)
                dataset_dict["bg_mask"] = (~all_masks).to(dtype = torch.uint8)

                dataset_dict["scrbs_count"] = dataset_dict["fg_scrbs"][0].shape[0] #+ dataset_dict["bg_scrbs"].shape[0]
                
                # print("gt_masks:",gt_masks.shape)
                # print("all_masks:",all_masks.shape)
                # gt_masks = gt_masks.tensor.unsqueeze(0)
                # fg_scrbs, bg_scrbs = generate_point_to_blob_masks_eval_deterministic(gt_masks, all_masks=all_masks, max_num_points=1)
                # dataset_dict["fg_scrbs"] = fg_scrbs.squeeze(0)
                
                # dataset_dict["bg_scrbs"] = None
                # dataset_dict["bg_mask"] = (~all_masks).to(dtype = torch.uint8)
                # dataset_dict["scrbs_count"] = dataset_dict["fg_scrbs"].shape[0] #+ dataset_dict["bg_scrbs"].shape[0]
            else:
                return None

            dataset_dict["instances"] = instances

        return dataset_dict
