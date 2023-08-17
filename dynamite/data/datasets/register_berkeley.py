import os
import torch
from detectron2.structures import BitMasks, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import numpy as np
from pycocotools import coco
from detectron2.structures import BoxMode
from .utils import get_labels_with_sizes, bbox_from_mask_np

def get_berkeley_dicts(data_dir):
    img_dir =  os.path.join(data_dir,"images/")
    gt_mask_dir = os.path.join(data_dir, "masks/")
    files_list = os.listdir(img_dir)
    img_list = [img_dir+x for x in files_list if '.jpg' in x]
    mask_dict = {}
    for i in img_list:
        mask_path = i.replace('.jpg','.png').replace('/images/','/masks/')
        mask_dict[i] = mask_path

    dataset_dicts = []
    for img_path in img_list:

        record = {}
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w = image.shape[:2]
        record['file_name'] = img_path
        record['height'] = h
        record['width'] = w
        record['image_id'] = img_path[:].split('/')[-1][:-4]

        mask_path = mask_dict[img_path]
        instances_mask = cv2.imread(mask_path).astype(np.uint8)
        
        if len(instances_mask.shape) == 3:
            instances_mask = instances_mask[:,:,0]
        ignore_mask = instances_mask==128
        ignore_mask = ignore_mask.astype(np.uint8)
        ignore_mask = torch.from_numpy(ignore_mask).unsqueeze(0)

        obj = {}
        instances_mask = instances_mask > 128
        obj_mask = instances_mask.astype(np.uint8)
        bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
        obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
            'iscrowd': 0, 'id': 1}
    
        obj["bbox"] = bbox
        obj["bbox_mode"] = BoxMode.XYXY_ABS
        record["annotations"] = [obj]

        record['ignore_mask'] =  ignore_mask

        dataset_dicts.append(record)
    return dataset_dicts

def _register_berkeley(data_dir):

    for d in [data_dir]:
        DatasetCatalog.register("Berkeley", lambda d=d: get_berkeley_dicts(d))

_data_dir = os.path.join(os.getcwd(),"datasets/Berkeley")        
_register_berkeley(_data_dir)
