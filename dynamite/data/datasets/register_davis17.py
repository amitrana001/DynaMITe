# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import glob
import logging
import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from PIL import Image
from pycocotools import coco
from imantics import Polygons, Mask
"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_davis", "register_davis_instances"]

# ==== Predefined splits for DAVIS 2017 ===========
_PREDEFINED_SPLITS_DAVIS_2017 = {
"davis_2017_val": ("DAVIS/DAVIS-2017-trainval/Annotations/480p",
"DAVIS/DAVIS-2017-trainval/JPEGImages/480p",
"DAVIS/DAVIS-2017-trainval/ImageSets/2017/val.txt"),
}

def register_all_davis17(root):
    for key, (ann_root, image_root, imset) in _PREDEFINED_SPLITS_DAVIS_2017.items():
        register_davis_instances(
        key,
        _get_davis_2017_instances_meta(),
        os.path.join(root, ann_root),
        os.path.join(root, image_root),
        os.path.join(root, imset)
    )
    # print("davis_2017_val datset registered")

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


def _get_davis_2017_instances_meta():
    return {}


def load_davis(annotation_root, image_root, imset, dataset_name=None, extra_annotation_keys=None):

    dataset_dicts = []
    with open(imset, "r") as lines:
        for _video_id, line in enumerate(lines):
            _video = line.rstrip('\n')
            img_list = np.array(glob.glob(os.path.join(image_root, _video, '*.jpg')))
            img_list.sort()

            # filter out empty annotations during training
            mask_list = np.array(glob.glob(os.path.join(annotation_root, _video, '*.png')))
            mask_list.sort()
            _mask_file = os.path.join(annotation_root, _video, '00000.png')
            _mask = np.array(Image.open(_mask_file).convert("P"))
            height, width = _mask.shape
            num_objects = np.max(_mask)
            for i, (_img_file, _mask_file) in enumerate(zip(img_list,mask_list)):
                record = {}
                record["file_name"] = _img_file
                record["height"] = height
                record["width"] = width
                record["image_id"] = _video + str(_video_id) + _img_file[-9:-4]

                frame_objs = []
                frame_mask = np.array(Image.open(_mask_file).convert("P")).astype(np.uint8)
                for obj_id in range(1, num_objects + 1):
                    obj = {} 
                    obj_mask = (frame_mask == obj_id).astype(np.uint8)
                    bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
                    obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
                        'iscrowd': False, 'id': obj_id}
                  
                    obj["bbox"] = bbox
                    obj["bbox_mode"] = BoxMode.XYXY_ABS
                    frame_objs.append(obj)
                record["annotations"] = frame_objs
                dataset_dicts.append(record)
    return dataset_dicts


def register_davis_instances(name, metadata, ann_root, image_root, imset):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_davis(ann_root, image_root, imset, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        ann_root=ann_root, image_root=image_root, imset=imset, evaluator_type="davis", **metadata
    )

_root = os.getcwd()
_root = os.path.join(_root, "datasets/")
register_all_davis17(_root)
