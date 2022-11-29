# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

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

from torchvision import transforms
from pycocotools import mask as coco_mask
from mask2former.data.scribble.gen_scribble import get_scribble_gt, get_scribble_gt_mask
# from coco_instance_interactive_dataset_mapper import filter_instances, build_transform_gen, convert_coco_poly_to_mask
from mask2former.data.points.annotation_generator import generate_point_to_blob_masks_eval, generate_point_to_blob_masks_eval_deterministic
from mask2former.data.scribble.gen_scribble import get_scribble_eval, get_scribble_gt_mask

__all__ = ["COCOMvalDatasetMapper"]

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

# This is specifically designed for the COCO dataset.
class COCOMvalDatasetMapper:
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

        # if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            # return dataset_dict

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
                new_instances = Instances(image_size=image_shape)
                all_masks = dataset_dict["padding_mask"].int()
                
                    
                new_instances.set('gt_masks', gt_masks)
                new_instances.set('gt_classes', instances.gt_classes)
                new_instances.set('gt_boxes', instances.gt_boxes) 
                   
                gt_masks = gt_masks.unsqueeze(0)
                fg_scrbs, bg_scrbs = generate_point_to_blob_masks_eval_deterministic(gt_masks, all_masks=all_masks, max_num_points=1)
                # print(gt_masks.shape)
                # fg = get_scribble_eval(np.asarray(gt_masks[0][0]).astype(np.uint8)*255, bg = False)
                # if np.argwhere(fg).shape[0] == 0:
                #     return None
                # fg_scribs = torch.from_numpy(fg)
                # dataset_dict["fg_scrbs"] = fg_scribs.unsqueeze(0)
                dataset_dict["fg_scrbs"] = fg_scrbs.squeeze(0)
                dataset_dict["bg_scrbs"] = None
                # dataset_dict["bg_scrbs"] = bg_scrbs
                # dataset_dict["bg_scrbs"] = torch.zeros_like(bg_scrbs)
                dataset_dict["bg_mask"] = (~all_masks).to(dtype = torch.uint8)

                dataset_dict["scrbs_count"] = dataset_dict["fg_scrbs"].shape[0] #+ dataset_dict["bg_scrbs"].shape[0]
            else:
                return None
            
            dataset_dict["instances"] = new_instances
            

        return dataset_dict
