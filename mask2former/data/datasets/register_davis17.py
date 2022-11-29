# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import glob
import logging
import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.timer import Timer
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
    print("davis_2017_val datset registered")

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
    timer = Timer()
    # image_dir = os.path.join(root, 'JPEGImages', '480p')
    # mask_dir = os.path.join(root, 'Annotations_unsupervised', '480p')
    from PIL import Image
    dataset_dicts = []
    with open(imset, "r") as lines:
        # for _video_id, line in enumerate(['judo']):
        for _video_id, line in enumerate(lines):
            _video = line.rstrip('\n')
            # print(_video)
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
                # record["length"] = len(img_list)
                # record['video_name'] = _video
                # video_id = record["video_id"] = _video_id
                record["image_id"] = _video + str(_video_id) + _img_file[-9:-4]

                # num_frames = len(img_list)
                
                video_objects = []
                
                frame_objs = []
                frame_mask = np.array(Image.open(_mask_file).convert("P")).astype(np.uint8)
                for obj_id in range(1, num_objects + 1):
                # for obj_id in range(1, num_objects + 1):
                    obj = {} 
                    obj_mask = (frame_mask == obj_id).astype(np.uint8)
                    bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
                    # polygons = Mask(coco.maqskUtils.encode(np.asfortranarray(obj_mask))).polygons()
                    obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
                        'iscrowd': False, 'id': obj_id}
                    # obj["segmentation"] = coco.maskUtils.encode(np.asfortranarray(obj_mask))
                    # obj['category_id'] =1
                    # obj['id'] = obj_id
                    # obj['iscrowd']: False
                
                    obj["bbox"] = bbox
                    obj["bbox_mode"] = BoxMode.XYXY_ABS
                    frame_objs.append(obj)
                # video_objects.append(frame_objs)
                record["annotations"] = frame_objs
                dataset_dicts.append(record)
            # record = {}
            # record["file_names"] = img_list
            # record["height"] = height
            # record["width"] = width
            # record["length"] = len(img_list)
            # record['video_name'] = _video
            # video_id = record["video_id"] = _video_id

            # num_frames = len(img_list)
            # num_objects = np.max(_mask)
            # video_objects = []
            # for i, _mask_file in enumerate(mask_list):
            #     frame_objs = []
            #     frame_mask = np.array(Image.open(_mask_file).convert("P")).astype(np.uint8)
            #     for obj_id in range(1, num_objects + 1):
            #     # for obj_id in range(1, num_objects + 1):
            #         obj_mask = (frame_mask == obj_id).astype(np.uint8)
            #         bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
            #         obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
            #                'iscrowd': False, 'id': obj_id}
            #         obj["bbox"] = bbox
            #         obj["bbox_mode"] = BoxMode.XYXY_ABS
            #         frame_objs.append(obj)
            #     video_objects.append(frame_objs)
            # record["annotations"] = video_objects
            # dataset_dicts.append(record)
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
    # print(image_root,ann_root)
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_davis(ann_root, image_root, imset, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        ann_root=ann_root, image_root=image_root, imset=imset, evaluator_type="davis", **metadata
    )

_root = os.getcwd()
_root = os.path.join(_root, "datasets/")
# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# print(_root)
register_all_davis17(_root)

# if __name__ == "__main__":
#     """
#     Test the Davis dataset loader.
#     """
#     from detectron2.utils.logger import setup_logger
#     from detectron2.utils.visualizer import Visualizer
#     import detectron2.data.datasets  # noqa # add pre-defined metadata
#     from PIL import Image

#     logger = setup_logger(name=__name__)
#     #assert sys.argv[3] in DatasetCatalog.list()
#     meta = MetadataCatalog.get("davis_2017")

#     image_root = "./datasets/davis/JPEGImages/480p"
#     ann_root = "./datasets/davis_point_vos/Annotations/480p"
#     imset = "./datasets/davis/ImageSets/2017/val.txt"
#     dicts = load_davis(ann_root, image_root, imset, dataset_name="davis_2017_val")
#     logger.info("Done loading {} samples.".format(len(dicts)))

#     dirname = "davis-vis"
#     os.makedirs(dirname, exist_ok=True)

#     def extract_frame_dic(dic, frame_idx):
#         import copy
#         frame_dic = copy.deepcopy(dic)
#         annos = frame_dic.get("annotations", None)
#         if annos:
#             frame_dic["annotations"] = annos[frame_idx]

#         return frame_dic

#     for d in dicts:
#         vid_name = d["file_names"][0].split('/')[-2]
#         os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
#         for idx, file_name in enumerate(d["file_names"]):
#             img = np.array(Image.open(file_name))
#             visualizer = Visualizer(img, metadata=meta)
#             vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
#             fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
#             vis.save(fpath)