import os
import torch
from detectron2.structures import BitMasks, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import numpy as np

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
        # print(mask_path)
        instances_mask = cv2.imread(mask_path).astype(np.uint8)
        if len(instances_mask.shape) == 3:
            instances_mask = instances_mask[:,:,0]
        ignore_mask = instances_mask==128
        ignore_mask = ignore_mask.astype(np.uint8)
        ignore_mask = torch.from_numpy(ignore_mask).unsqueeze(0)

        instances_mask = instances_mask > 128
        instances_mask = instances_mask.astype(np.uint8)
        instances_mask = torch.from_numpy(instances_mask).unsqueeze(0)
        gt_classes = torch.tensor([1])
        boxes = Boxes(torch.tensor([[0.,0.,0.,0.]]))
        inst = Instances(image_size=(h,w))
        inst.set('gt_masks', instances_mask)
        inst.set('gt_classes', gt_classes)
        inst.set('gt_boxes', boxes)
        record['instances'] = inst
        record['ignore_mask'] =  ignore_mask

        dataset_dicts.append(record)
    return dataset_dicts

def register_berkeley():

    things_classes =  MetadataCatalog.get("coco_2017_train").thing_classes

    data_dir = os.path.join(os.getcwd(),"datasets/Berkeley")
    # data_dir = "/home/rana/claix_work/InteractiveM2F/datasets/Berkeley/"
    for d in [data_dir]:
        DatasetCatalog.register("Berkeley", lambda d=d: get_berkeley_dicts(d))
        MetadataCatalog.get("Berkeley").set(thing_classes=things_classes)
    print("Berkeley data registered")

register_berkeley()