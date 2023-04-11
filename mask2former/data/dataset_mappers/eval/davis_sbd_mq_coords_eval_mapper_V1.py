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
from detectron2.structures import BitMasks, Instances, Boxes
from detectron2.structures.masks import PolygonMasks

from mask2former.data.dataset_mappers.mapper_utils.datamapper_utils import  build_transform_gen

from mask2former.evaluation.eval_utils import get_gt_clicks_coords_eval

__all__ = ["DAVISSBDMQCoordsV1EvalMapper"]

# This is specifically designed for the COCO dataset.
class DAVISSBDMQCoordsV1EvalMapper:
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
        orig_image_shape = image.shape[:2]
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
                utils.transform_instance_annotations(copy.deepcopy(obj), transforms, image_shape)
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
            
            # orig_annos = [
            #     utils.transform_instance_annotations(obj, None, orig_image_shape)
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]
            # orig_instances = utils.annotations_to_instances(orig_annos, orig_image_shape,  mask_format="bitmask")
            # orig_instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # # boxes_area = instances.gt_boxes.area()
            # # Need to filter empty instances first (due to augmentation)
            # orig_instances = utils.filter_empty_instances(instances)

            # dataset_dict["orig_gt_masks"] = orig_instances.gt_masks.tensor
            
            
            if len(instances) == 0:
                # print("here")
                return None
            # Generate masks from polygon
            h, w = instances.image_size
        
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks.tensor
                # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)

                mask_areas = torch.sum(gt_masks, (1,2))
                gt_masks = gt_masks.to(dtype=torch.uint8)
                gt_masks =  gt_masks[sorted(range(len(mask_areas)),key=mask_areas.__getitem__,reverse=True)]

                instance_map = torch.zeros((gt_masks.shape[-2:]), dtype=torch.int16)
                num_objects = gt_masks.shape[0]
                instances_ids = np.arange(1, num_objects + 1)

                for _id, _m in enumerate(gt_masks):
                    instance_map[_m == 1] = _id+1
                    assert (_m != 0).sum() > 0
                
                new_gt_masks = []
                for _id in instances_ids:
                    _m = (instance_map == _id).to(dtype=torch.uint8)
                    if _m.sum() > 0:
                        new_gt_masks.append(_m)
                
                if not len(new_gt_masks):
                    return None
                new_gt_masks = torch.stack(new_gt_masks,dim=0)
                assert num_objects == new_gt_masks.shape[0]
                dataset_dict['semantic_map'] = instance_map
                
                # instances.gt_masks = gt_masks.tensor
                new_gt_classes = [0]*new_gt_masks.shape[0]
                # new_gt_boxes = instances.gt_masks.get_bounding_boxes()[random_indices]
                new_gt_boxes =  Boxes((np.zeros((new_gt_masks.shape[0],4))))
                
                new_instances = Instances(image_size=image_shape)
                new_instances.set('gt_masks', new_gt_masks)
                new_instances.set('gt_classes', new_gt_classes)
                new_instances.set('gt_boxes', new_gt_boxes) 
                
                all_masks = dataset_dict["padding_mask"].int()
                for gt_mask in new_gt_masks:
                    all_masks = torch.logical_or(all_masks, gt_mask)

                # gt_masks = gt_masks.tensor
                (num_scrbs_per_mask, fg_coords_list, bg_coords_list,
                fg_point_masks, bg_point_masks) = get_gt_clicks_coords_eval(new_gt_masks, unique_timestamp = self.unique_timestamp)
        
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

            dataset_dict["instances"] = new_instances

        return dataset_dict
