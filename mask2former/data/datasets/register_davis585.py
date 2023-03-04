import os
import torch
from detectron2.structures import BitMasks, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import numpy as np

def get_davis585_dicts(data_dir):

    # print("here")
    sequence_names = os.listdir(data_dir)
    dataset_dicts = []
    for sequence_name in sequence_names:
        sequence_dir = data_dir + sequence_name + '/'
        gt_names = os.listdir(sequence_dir)
        gt_names = [i  for i in gt_names if '.png' in i and 'init' not in i]
        for gt_name in gt_names:
            record = {}
            mask_path = sequence_dir + gt_name
            
            image_name = gt_name.split('_')[-1].replace('.png','.jpg')
            image_path = sequence_dir + image_name
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h,w = image.shape[:2]
            record['file_name'] = image_path
            record['height'] = h
            record['width'] = w
            record['image_id'] = image_path[:]

            stm_init_name = 'init_stm_' + gt_name
            stm_init_path = sequence_dir + stm_init_name

            sp_init_name = 'init_sp_' + gt_name
            sp_init_path = sequence_dir + sp_init_name

            instances_mask = cv2.imread(mask_path)[:,:,0] > 128
            stm_init_mask = cv2.imread(stm_init_path)[:,:,0] > 128
            sp_init_mask = cv2.imread(sp_init_path)[:,:,0] > 128
            record['stm_init_mask'] = torch.from_numpy(stm_init_mask).to(dtype=torch.uint8)
            record['sp_init_mask']  = torch.from_numpy(sp_init_mask).to(dtype=torch.uint8)

            instances_mask = instances_mask.astype(np.uint8)
            instances_mask = torch.from_numpy(instances_mask).unsqueeze(0)
            gt_classes = torch.tensor([1])
            boxes = Boxes(torch.tensor([[0.,0.,0.,0.]]))
            inst = Instances(image_size=(h,w))
            inst.set('gt_masks', instances_mask)
            inst.set('gt_classes', gt_classes)
            inst.set('gt_boxes', boxes)
            record['instances'] = inst

            dataset_dicts.append(record)
    return dataset_dicts

def register_davis585():

    # things_classes =  MetadataCatalog.get("coco_2017_train").thing_classes

    data_dir = os.path.join(os.getcwd(),"datasets/DAVIS585/data/")
    # data_dir = "/home/rana/claix_work/InteractiveM2F/datasets/COCO_MVal/"
    for d in [data_dir]:
        DatasetCatalog.register("davis585", lambda d=d: get_davis585_dicts(d))
        MetadataCatalog.get("davis585").set(thing_classes=None)
    # print("COCO MVal data registered")

register_davis585()