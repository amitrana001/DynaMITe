import os
import torch
from detectron2.structures import BitMasks, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import numpy as np
from scipy.io import loadmat
from pycocotools import coco
from detectron2.structures import BoxMode

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

def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()

def get_sbd_single_inst_dicts(data_dir,split="val"):

    img_dir =  os.path.join(data_dir,"img/")
    gt_mask_dir = os.path.join(data_dir, "inst/")

    val_txt_path = os.path.join(data_dir, f'{split}.txt')

    with open(val_txt_path, 'r') as f:
        dataset_samples = [x.strip() for x in f.readlines()]
    
    dataset_dicts = []
    for sample in dataset_samples:
        
        img_path = os.path.join(img_dir, f'{sample}.jpg')
        mask_path = os.path.join(gt_mask_dir, f'{sample}.mat')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w = image.shape[:2]
        instances_mask = loadmat(str(mask_path))['GTinst'][0][0][0].astype(np.int32)
        instances_ids, _ = get_labels_with_sizes(instances_mask)
        for obj_id in instances_ids:

            record = {}
            
            record['file_name'] = img_path
            record['height'] = h
            record['width'] = w
            record['image_id'] = sample

            obj = {} 
            obj_mask = (instances_mask == obj_id).astype(np.uint8)
            bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
            obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
                'iscrowd': False, 'id': obj_id}
        
            obj["bbox"] = bbox
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            record["annotations"] = [obj]
            dataset_dicts.append(record)
       
    return dataset_dicts

def _register_sbd_single_inst(data_dir):

    for d in [data_dir]:
        DatasetCatalog.register("sbd_single_inst", lambda d=d: get_sbd_single_inst_dicts(d))

_data_dir = os.path.join(os.getcwd(),"datasets/sbd/dataset")
_register_sbd_single_inst(_data_dir)
