# ------------------------------------------------------------------------
# Semantic SAM
# Copyright (c) MicroSoft, Inc. and its affiliates.
# Modified by Xueyan Zou and Jianwei Yang.
# ------------------------------------------------------------------------

import copy
import json
import logging
import os
import numpy as np
import torch
import random

from detectron2.structures import Instances, Boxes, PolygonMasks,BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask
# from ..utils.tsv
# from ..utils.tsv import TSVFile, img_from_base64, generate_lineidx, FileProgressingbar
from detectron2.config import configurable
from dynamite.data.dataset_mappers.utils import get_clicks_coords, build_transform_gen

__all__ = ["SamBaselineDatasetMapperJSON"]

# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2021-08-13 14:35:27
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-04-24 11:38:58

import os
import base64
from io import BytesIO
from PIL import Image

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    image = BytesIO(jpgbytestring)
    image = Image.open(image).convert("RGB")
    return image

    # jpgbytestring = base64.b64decode(imagestring)
    # nparr = np.frombuffer(jpgbytestring, np.uint8)
    # try:
    #     r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #     # r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    #     return r
    # except:
    #     return None


def generate_lineidx(filein, idxout):
    assert not os.path.isfile(idxout)
    with open(filein, 'r') as tsvin, open(idxout, 'w') as tsvout:
        bar = FileProgressingbar(tsvin, 'Generating lineidx {0}: '.format(idxout))
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
            bar.update()
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
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

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if not is_train:
        return T.ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            ),
    # assert is_train, "Only support training augmentation"
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentation = []

    if cfg_input['RANDOM_FLIP'] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


class SamBaselineDatasetMapperJSON:
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
        augmentation,
        image_format,
    ):
        self.augmentation = augmentation
        logging.getLogger(__name__).info(
            "[COCO_Instance_LSJ_Augment_Dataset_Mapper] Full TransformGens used in training: {}".format(str(self.augmentation))
        )
        # _root = os.getenv("SAM_DATASETS", "datasets")

        totoal_images = 0

        self.img_format = image_format
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "augmentation": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
        }
        return ret
    
    def read_img(self, row):
        img = img_from_base64(row[-1])
        return img

    def read_json(selfself, row):
        anno=json.loads(row[1])
        return anno

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["img_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        ori_shape = image.shape[:2]

        # image, transforms = T.apply_transform_gens(self.augmentation, image)
        # image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        anns = json.load(open(dataset_dict["ann_name"], 'r'))['annotations']
        dataset_dict['annotations'] = anns
        
        for anno in dataset_dict['annotations']:
            anno["bbox_mode"] = BoxMode.XYWH_ABS
            anno["category_id"] = 0

        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.augmentation, image)

        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        
        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            mask_shape = ori_shape
            if len(dataset_dict['annotations'])>0 and 'segmentation' in dataset_dict['annotations'][0].keys() and 'size' in dataset_dict['annotations'][0]['segmentation'].keys():
                mask_shape = dataset_dict['annotations'][0]['segmentation']['size']
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape,mask_format='bitmask')
            
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'): 
                return None
                # instances.gt_masks = PolygonMasks([])  # for negative examples
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)

            instances = utils.filter_empty_instances(instances)
            if len(instances) == 0:
                return None
            # Generate masks from polygon
            h, w = instances.image_size

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

        