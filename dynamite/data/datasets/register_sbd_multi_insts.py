import os
import torch
from detectron2.structures import BitMasks, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import numpy as np
from scipy.io import loadmat
from pycocotools import coco
from detectron2.structures import BoxMode
from .utils import bbox_from_mask_np, get_labels_with_sizes

def get_sbd_multi_insts_dicts(data_dir,split="val"):

    img_dir =  os.path.join(data_dir,"img/")
    gt_mask_dir = os.path.join(data_dir, "inst/")

    val_txt_path = os.path.join(data_dir, f'{split}.txt')

    with open(val_txt_path, 'r') as f:
        dataset_samples = [x.strip() for x in f.readlines()]
    
    dataset_dicts = []
    for sample in dataset_samples:

        img_path = os.path.join(img_dir, f'{sample}.jpg')
        mask_path = os.path.join(gt_mask_dir, f'{sample}.mat')
        record = {}
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w = image.shape[:2]
        record['file_name'] = img_path
        record['height'] = h
        record['width'] = w
        record['image_id'] = sample

        instances_mask = loadmat(str(mask_path))['GTinst'][0][0][0].astype(np.int32)
        instances_ids, _ = get_labels_with_sizes(instances_mask)
        frame_objs = []
        for obj_id in instances_ids:
            obj = {} 
            obj_mask = (instances_mask == obj_id).astype(np.uint8)
            bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
            obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
                'iscrowd': False, 'id': obj_id}
        
            obj["bbox"] = bbox
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            frame_objs.append(obj)
        record["annotations"] = frame_objs
        dataset_dicts.append(record)
    return dataset_dicts

def _register_sbd_multi_insts(data_dir):    
   
    for d in [data_dir]:
        DatasetCatalog.register("sbd_multi_insts", lambda d=d: get_sbd_multi_insts_dicts(d))

_data_dir = os.path.join(os.getcwd(),"datasets/sbd/dataset")        
_register_sbd_multi_insts(_data_dir)
