# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import pickle
import numpy as np
import torch
import random
import cv2
from copy import deepcopy
from detectron2.structures import BoxMode
from fvcore.common.timer import Timer
from pycocotools import coco
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances
from detectron2.structures.masks import PolygonMasks

from pycocotools import mask as coco_mask
from mask2former.data.scribble.gen_scribble import get_scribble_eval, get_scribble_gt_mask
from mask2former.data.dataset_mappers.mapper_utils.datamapper_utils import convert_coco_poly_to_mask, filter_instances, build_transform_gen

from mask2former.data.points.annotation_generator import gen_multi_points_per_mask, generate_point_to_blob_masks

__all__ = ["COCOLVISMultiInstMQClicksDatasetMapper"]


def bbox_from_mask_np(mask, order='Y1Y2X1X2', return_none_if_invalid=False):
  if len(np.where(mask)[0]) == 0:
    return np.array([-1, -1, -1, -1])
  x_min = np.where(mask)[1].min()
  x_max = np.where(mask)[1].max()

  y_min = np.where(mask)[0].min()
  y_max = np.where(mask)[0].max()

  if order == 'Y1Y2X1X2':
    return np.array([y_min, y_max, x_min, x_max])
  elif order == 'X1X2Y1Y2':
    return np.array([x_min, x_max, y_min, y_max])
  elif order == 'X1Y1X2Y2':
    return np.array([x_min, y_min, x_max, y_max])
  elif order == 'Y1X1Y2X2':
    return np.array([y_min, x_min, y_max, x_max])
  else:
    raise ValueError("Invalid order argument: %s" % order)

def sample_from_masks_layer(instances_info, encoded_masks, obj_id):
    # objs_tree = sample._objects
    # objs_tree = instances_info

    node_mask = get_object_mask(instances_info, encoded_masks, obj_id)
    gt_mask = node_mask
    return gt_mask, [node_mask], []

def get_object_mask(instances_info, encoded_masks, obj_id):

    layer_indx, mask_id = instances_info[obj_id]['mapping']
    obj_mask = (encoded_masks[:, :, layer_indx] == mask_id).astype(np.int32)
    # if self._ignored_regions:
    #     for layer_indx, mask_id in self._ignored_regions:
    #         ignore_mask = self._encoded_masks[:, :, layer_indx] == mask_id
    #         obj_mask[ignore_mask] = -1

    return obj_mask


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
class COCOLVISMultiInstMQClicksDatasetMapper:
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
        self.min_area = 1000.0
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

        stuff_prob = 0.15
        if "annotations" in dataset_dict:

            # USER: Implement additional transformations if you have other types of data
            if stuff_prob > 0 and random.random() < stuff_prob: 
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
            else:
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if (obj.get("iscrowd", 0) == 0 and obj.get("isThing"))
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
            instances = filter_coco_lvis_instances(instances, min_area = 1000.0)
            # print(f"instances after filter:{instances.gt_masks.tensor.shape}")
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
                gt_masks = instances.gt_masks.tensor.to(dtype=torch.uint8)
                # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                # instances.gt_masks = gt_masks.tensor

                all_masks = dataset_dict["padding_mask"].int()
                # filterd_gt_masks = []
                new_instances = Instances(image_size=image_shape)
                    # if gt_masks.shape[0] == 0:
                    #     return None

                if gt_masks.shape[0] == 1:
                    num_masks = 1
                else:
                    #Take 75% masks as the foreground masks
                    num_masks = int(gt_masks.shape[0]*(0.75))
                # num_masks = torch.randint(1, gt_masks.shape[0]+1, (1,))[0]
                # print(f'gt_masks:{gt_masks.shape[0]}, num_masks:{num_masks}')
                random_indices = random.sample(range(gt_masks.shape[0]),num_masks)
                new_gt_masks = gt_masks[random_indices]
                new_gt_classes = instances.gt_classes[random_indices]
                new_gt_boxes = instances.gt_masks.get_bounding_boxes()[random_indices]
                # instances.gt_masks = gt_masks[random_indices]
                # instances.gt_classes = instances.gt_classes[random_indices]
                new_instances.set('gt_masks', new_gt_masks)
                new_instances.set('gt_classes', new_gt_classes)
                new_instances.set('gt_boxes', new_gt_boxes) 
                # filterd_gt_masks = []
                # print(random_indices)
                for m in new_gt_masks:
                    all_masks = torch.logical_or(all_masks, m)
                
                # new_gt_masks = new_gt_masks.unsqueeze(0)
                # print(new_gt_masks.shape)
                points_masks = gen_multi_points_per_mask(new_gt_masks, all_masks=all_masks)
                if points_masks is None:
                    return None
                else:
                    fg_masks, bg_masks, num_scrbs_per_mask= points_masks
                dataset_dict["fg_scrbs"] = fg_masks
                    # only_bg_mask =  get_scribble_gt_mask(np.asarray(all_masks).astype(np.uint8)*255, bg = True)
                    
                dataset_dict["bg_scrbs"] = bg_masks
                dataset_dict["num_scrbs_per_mask"] = num_scrbs_per_mask
                # print(masks.tensor.dtype)
                assert len(num_scrbs_per_mask) == new_instances.gt_masks.shape[0]
                if self.random_bg_queries:
                    pick = np.random.rand()
                    if pick < 0.20:
                        dataset_dict["bg_scrbs"] = None
                if dataset_dict['bg_scrbs'] is None:
                    dataset_dict["scrbs_count"] = len(dataset_dict["fg_scrbs"])
                else:
                    dataset_dict["scrbs_count"] = len(dataset_dict["fg_scrbs"]) + len(dataset_dict["bg_scrbs"])
            else:
                return None

            dataset_dict["instances"] = new_instances

        return dataset_dict
