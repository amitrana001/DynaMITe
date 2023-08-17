# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import glob
import logging
import os
import random
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.timer import Timer
from pycocotools import coco
from imantics import Polygons, Mask
import pickle
import cv2
from copy import deepcopy
"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_coco_lvis", "register_all_coco_lvis_2017"]

# ==== Predefined splits for coco_lvis 2017 ===========

_PREDEFINED_SPLITS_COCO_LVIS_2017 = {
"coco_lvis_2017_train": ("lvis/train/hannotation.pickle",
"lvis/train/images",
"lvis/train/masks",
"lvis/coco_lvis_combined.pickle"),
}


def register_all_coco_lvis_2017(root):
    for key, (ann_root, image_root, imset, data_pickle_file) in _PREDEFINED_SPLITS_COCO_LVIS_2017.items():
        register_coco_lvis_2017_instances(
        key,
        _get_coco_lvis_2017_instances_meta(),
        os.path.join(root, ann_root),
        os.path.join(root, image_root),
        os.path.join(root, imset),
        os.path.join(root, data_pickle_file),
    )
    # print("coco_lvis_2017_train datset registered")

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

    node_mask = get_object_mask(instances_info, encoded_masks, obj_id)
    gt_mask = node_mask
    return gt_mask, [node_mask], []

def get_object_mask(instances_info, encoded_masks, obj_id):

    layer_indx, mask_id = instances_info[obj_id]['mapping']
    obj_mask = (encoded_masks[:, :, layer_indx] == mask_id).astype(np.int32)

    return obj_mask

def _get_coco_lvis_2017_instances_meta():
    return {}

def load_coco_lvis(annotation_root, image_root, masks_root, pickle_file_path, stuff_prob=1.0, extra_annotation_keys=None):

    if os.path.exists(pickle_file_path):
        print(f"Found pickle file:{pickle_file_path}")
        with open(pickle_file_path, 'rb') as f:
            dataset_dicts = pickle.load(f)
        return dataset_dicts

    with open(annotation_root, 'rb') as f:
        dataset_samples = sorted(pickle.load(f).items())

    dataset_dicts = []
    
    for index in range(0, len(dataset_samples)):
        record = {}
        image_id, sample = dataset_samples[index]
        image_path = image_root + f'/{image_id}.jpg'

        image = cv2.imread(image_path)
        h,w,_ = image.shape
        
        packed_masks_path = masks_root + f'/{image_id}.pickle'
        
        record['image_id'] = image_id
        record["file_name"] = image_path
        record['height'] = h
        record['width'] = w

        with open(packed_masks_path, 'rb') as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)

        instances_info = deepcopy(sample['hierarchy'])

        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {'children': [], 'parent': None, 'node_level': 0}
                instances_info[inst_id] = inst_info
            inst_info['mapping'] = objs_mapping[inst_id]
       
        things_obj_ids = [obj_id for obj_id, obj_info in instances_info.items() if obj_info['parent'] is None]

        for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
            instances_info[inst_id] = {
                'mapping': objs_mapping[inst_id],
                'parent': None,
                'children': []
            }
        
        root_objects = [obj_id for obj_id, obj_info in instances_info.items() if obj_info['parent'] is None]

        objs = []
        for obj_id in root_objects:
            obj_gt_mask, obj_pos_segments, obj_neg_segments = sample_from_masks_layer(instances_info, layers, obj_id)

            obj = {} 
            obj_mask = obj_gt_mask.astype(np.uint8)
            obj_area = obj_mask.sum()
            bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
           
            obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
                'iscrowd': 0, 'id': obj_id, 'isThing': obj_id in things_obj_ids, 'area': obj_area}

            obj["bbox"] = bbox
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    with open(pickle_file_path, 'wb') as handle:
        pickle.dump(dataset_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset_dicts

def register_coco_lvis_2017_instances(name, metadata, ann_root, image_root, imset, pickle_file_path):
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
    DatasetCatalog.register(name, lambda: load_coco_lvis(ann_root, image_root, imset, pickle_file_path))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        ann_root=ann_root, image_root=image_root, imset=imset, evaluator_type="lvis", **metadata
    )

_root = "datasets/"
register_all_coco_lvis_2017(_root)
