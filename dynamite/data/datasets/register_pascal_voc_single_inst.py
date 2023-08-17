import os
import torch
from detectron2.structures import BitMasks, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import numpy as np
from pycocotools import coco
from detectron2.structures import BoxMode
from .utils import get_labels_with_sizes, bbox_from_mask_np

def get_pascal_voc_dicts(data_dir, split="val"):
    img_dir =  os.path.join(data_dir,"JPEGImages/")
    gt_mask_dir = os.path.join(data_dir, "SegmentationObject/")
    # files_list = os.listdir(img_dir)
    # img_list = [img_dir+x for x in files_list if '.jpg' in x]

    with open(os.path.join(data_dir,f'ImageSets/Segmentation/{split}.txt'), 'r') as f:
        dataset_samples = [name.strip() for name in f.readlines()]
    
    dataset_dicts = []
    for i, sample_id in enumerate(dataset_samples):
        image_path = str(os.path.join(img_dir,f'{sample_id}.jpg'))
        mask_path = str(os.path.join(gt_mask_dir,f'{sample_id}.png'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w = image.shape[:2]
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)

        objects_ids = np.unique(instances_mask)
        objects_ids = [x for x in objects_ids if x != 0 and x != 220]

        for instance_id in objects_ids:
            record = {}
           
            record['file_name'] = image_path
            record['height'] = h
            record['width'] = w
            record['image_id'] = sample_id

            ignore_mask = (instances_mask==220)
            ignore_mask = ignore_mask.astype(np.uint8)
            ignore_mask = torch.from_numpy(ignore_mask).unsqueeze(0)

            obj = {}
            obj_mask = (instances_mask == instance_id)
            obj_mask = obj_mask.astype(np.uint8)
            bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
            obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
                'iscrowd': 0, 'id': 1}
        
            obj["bbox"] = bbox
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            record["annotations"] = [obj]

            record['ignore_mask'] =  ignore_mask

            dataset_dicts.append(record)
    return dataset_dicts

def _register_pascal_voc(data_dir):

    for d in [data_dir]:
        DatasetCatalog.register("pascal_voc_single", lambda d=d: get_pascal_voc_dicts(d))

_data_dir = os.path.join(os.getcwd(),"datasets/pascal_voc")        
_register_pascal_voc(_data_dir)
