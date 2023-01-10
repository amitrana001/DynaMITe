# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
from threading import Thread

import numpy as np
import torch
import random
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances,Boxes
from detectron2.structures.masks import PolygonMasks
import torchvision.transforms.functional as F

from pycocotools import mask as coco_mask
from mask2former.data.scribble.gen_scribble import get_scribble_gt, get_scribble_gt_mask
# from coco_instance_interactive_dataset_mapper import filter_instances, build_transform_gen, convert_coco_poly_to_mask
from mask2former.data.points.annotation_generator import generate_point_to_blob_masks
from mask2former.data.points.annotation_generator import gen_multi_points_per_mask

from mask2former.data.dataset_mappers.mapper_utils.datamapper_utils import convert_coco_poly_to_mask, filter_instances, build_transform_gen

__all__ = ["COCOSingleInstMultiQueriesStuffClicksDatasetMapper"]

# This is specifically designed for the COCO dataset.
class COCOSingleInstMultiQueriesStuffClicksDatasetMapper:
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

        # distractor_objects= False
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
        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # apply the same transformation to panoptic segmentation
        # print(pan_seg_gt.shape)
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
        
        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)
        # pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        # pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)

        classes = []
        masks = []
        isThings = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if (not segment_info["iscrowd"]) and (segment_info["area"] > 1000):
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])
                isThings.append(segment_info['isthing'])
        isThings = np.asarray(isThings)
        if len(isThings)==0:
            return None
        things_indices = np.where(isThings)[0]
        stuff_indices = np.where(~isThings)[0]
        
        if (not things_indices.shape[0]) and (not stuff_indices.shape[0]):
            return None
        indx=-1
        while(indx == -1):
            pick = np.random.rand()
            if pick > 0.20:
                if things_indices.shape[0] != 0:
                    indx = np.random.choice(things_indices,1)
            else:
                if stuff_indices.shape[0] != 0:
                    indx = np.random.choice(stuff_indices,1)
        # print(indx)
        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes[indx], dtype=torch.int64)
        all_masks = dataset_dict["padding_mask"].int()
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            return None
            # instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
        else:
            # masks = BitMasks(
            #     torch.stack([torch.from_numpy(np.ascontiguousarray((masks[indx[0]].copy()))).to(dtype =torch.uint8)])
            # )
            # masks = torch.from_numpy(np.ascontiguousarray((masks[indx[0]].copy()))).to(dtype =torch.uint8)
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray((masks[i].copy()))).to(dtype =torch.uint8) for i in indx])
            )
            # print(f'masks:{masks.tensor.shape}')
            # masks = masks.unsqueeze(0)
            for m in masks:
                all_masks = torch.logical_or(all_masks, m)
            instances.gt_masks = masks.tensor.to(dtype=torch.uint8) #.tensor
            # instances.gt_boxes = masks.get_bounding_boxes()
            instances.gt_boxes = Boxes(torch.zeros((1, 4)))
            # mask = torch.from_numpy(np.ascontiguousarray(masks[indx[0]].copy())).to(dtype=torch.uint8).unsqueeze(0)
            
            points_masks = gen_multi_points_per_mask(masks.tensor, all_masks=all_masks)
            if points_masks is None:
                return None
            else:
                fg_masks, bg_masks, num_scrbs_per_mask= points_masks
            dataset_dict["fg_scrbs"] = fg_masks
                # only_bg_mask =  get_scribble_gt_mask(np.asarray(all_masks).astype(np.uint8)*255, bg = True)
                
            dataset_dict["bg_scrbs"] = bg_masks
            dataset_dict["num_scrbs_per_mask"] = num_scrbs_per_mask
            # print(masks.tensor.dtype)
            assert len(num_scrbs_per_mask) == instances.gt_masks.shape[0]
            if self.random_bg_queries:
                pick = np.random.rand()
                if pick < 0.20:
                    dataset_dict["bg_scrbs"] = None
            if dataset_dict['bg_scrbs'] is None:
                dataset_dict["scrbs_count"] = len(dataset_dict["fg_scrbs"])
            else:
                dataset_dict["scrbs_count"] = len(dataset_dict["fg_scrbs"]) + len(dataset_dict["bg_scrbs"])
        dataset_dict["instances"] = instances

        return dataset_dict

        
