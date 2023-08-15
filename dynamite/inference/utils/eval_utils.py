import os
import sys
import csv

import numpy as np
import torch, torchvision
import cv2
import logging
import datetime
import pickle
from prettytable import PrettyTable
from detectron2.utils.visualizer import Visualizer
import torchvision.transforms.functional as F
from dynamite.data.dataset_mappers.utils import create_circular_mask, get_max_dt_point_mask
# from offline_summary import get_statistics

def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))
color_map = get_palette(80)[1:]

def get_gt_clicks_coords_eval(masks, image_shape, max_num_points=1, ignore_masks=None,
                             first_click_center=True, t= 0):

    """
    :param masks: numpy array of shape I x H x W
    :param patch_size: size of patch (int)
    """
    # assert all_masks is not None
    masks = np.asarray(masks).astype(np.uint8)
    if ignore_masks is not None:
        not_ignores_mask = np.logical_not(np.asarray(ignore_masks, dtype=np.bool_))

    I, H, W = masks.shape
    trans_h, trans_w = image_shape
    ratio_h = trans_h/H
    ratio_w = trans_w/W
    num_clicks_per_object = [0]*I
    orig_fg_coords_list = []
    fg_coords_list = []
    # fg_point_masks = []
    for i, (_m) in enumerate(masks):
        orig_coords = []
        coords = []
        # point_masks_per_obj = []
        if first_click_center:
            if ignore_masks is not None:
                _m = np.logical_and(_m, not_ignores_mask[i]).astype(np.uint8)
            center_coords = get_max_dt_point_mask(_m, max_num_pts=max_num_points)
           
            orig_coords.append([center_coords[0], center_coords[1], t])
            coords.append([center_coords[0]*ratio_h, center_coords[1]*ratio_w, t])
            # if unique_timestamp:
            t+=1
            num_clicks_per_object[i]+=1
        orig_fg_coords_list.append(orig_coords) 
        fg_coords_list.append(coords)
       
    return num_clicks_per_object, fg_coords_list, orig_fg_coords_list


def log_single_instance(res, max_interactions, dataset_name, iou_threshold = 0.80):
    logger = logging.getLogger(__name__)
    total_num_instances = sum(res['total_num_instances'])
    total_num_interactions = sum(res['total_num_interactions'])
    num_failed_objects = sum(res['num_failed_objects'])
    total_iou = sum(res['total_iou'])

    logger.info(
        "Total number of instances: {}, Average num of interactions:{}".format(
            total_num_instances, total_num_interactions / total_num_instances
        )
    )
    logger.info(
        "Total number of failed cases: {}, Avg IOU: {}".format(
            num_failed_objects, total_iou / total_num_instances
        )
    )

    # 'res['dataset_iou_list'] is a list of dicts which has to be merged into a single dict
    dataset_iou_list = {}
    for _d in res['dataset_iou_list']:
        dataset_iou_list.update(_d)
    
    assert total_num_instances == len(dataset_iou_list)
    
    field_names = ["dataset"]
    row = [dataset_name]
    iou = 0.80
    while round(iou,2)<=iou_threshold:
        NOC, NFO, IOU = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=iou)
        field_names.extend([f"NoC_{int(iou*100)}", f"NFO_{int(iou*100)}"])
        row.extend([NOC, NFO])
        iou+=0.05

    field_names.extend(["#samples","#clicks"])
    row.extend([total_num_instances, max_interactions])

    table = PrettyTable()
    table.field_names = field_names
    table.add_row(row)
    print(table)


def log_multi_instance(res, dataset_name, iou_threshold=0.85, max_interactions = 10):
    logger = logging.getLogger(__name__)
    total_num_instances = sum(res['total_num_instances'])
    total_num_interactions = sum(res['total_num_interactions'])
   
    logger.info(
        "Total number of instances: {}, Average num of interactions:{}".format(
            total_num_instances, total_num_interactions / total_num_instances
        )
    )

    summary_stats = {}
    summary_stats["dataset"] = dataset_name
    summary_stats["iou_threshold"] = iou_threshold
    summary_stats["max_interactions"] = max_interactions
    num_interactions_per_image = {}
    for _d in res['num_interactions_per_image']:
        num_interactions_per_image.update(_d)
    final_iou_per_object = {}
    for _d in res['final_iou_per_object']:
        final_iou_per_object.update(_d)
    summary_stats["final_iou_per_object"] = final_iou_per_object
    summary_stats["num_interactions_per_image"] = num_interactions_per_image
    get_multi_inst_metrices(summary_stats)

def get_summary(dataset_iou_list, max_clicks=20, iou_thres=0.85):

    num_images =len(dataset_iou_list)
    total_clicks = 0
    failed_objects = 0
    total_iou = 0
    for (key, per_image_iou_list) in dataset_iou_list.items():
        vals = per_image_iou_list>=iou_thres
        if np.any(vals):
            num_clicks =  np.argmax(vals) + 1
            # total_iou += per_image_iou_list[num_clicks-1]
        else:
            assert len(vals) == max_clicks
            num_clicks =  max_clicks
            # total_iou += per_image_iou_list[-1]
            failed_objects+=1
        total_iou += per_image_iou_list[num_clicks-1]
        total_clicks+=num_clicks
    
    return np.round(total_clicks/num_images,2), failed_objects, np.round(total_iou/num_images,4)

def get_multi_inst_metrices(summary_stats):
   
    dataset_name = summary_stats["dataset"]
    iou_threshold = summary_stats["iou_threshold"]
    max_interactions = summary_stats["max_interactions"]
    
    final_iou_per_object = summary_stats["final_iou_per_object"]
    num_interactions_per_image = summary_stats["num_interactions_per_image"]
  
    NFO = 0
    NFI = 0
    NCI_all = 0.0
    NCI_suc = 0.0
    Avg_IOU = 0.0
    total_images = len(list(final_iou_per_object.keys()))
    total_num_instances = 0
    for _image_id in final_iou_per_object.keys():

        final_ious = final_iou_per_object[_image_id][-1]
        total_interactions_per_image = num_interactions_per_image[_image_id]

        Avg_IOU += sum(final_ious)/len(final_ious)
        NCI_all += total_interactions_per_image/len(final_ious)
        total_num_instances += len(final_ious)

        _is_failed_image = False
        _suc = 0
        for i, iou in enumerate(final_ious):
            if iou<iou_threshold:
                _is_failed_image = True
                NFO +=1
            else:
                _suc+=1
        if _suc!=0:
            NCI_suc += total_interactions_per_image/_suc
        else:
            NCI_suc += total_interactions_per_image
        if _is_failed_image:
            NFI+=1

    NCI_all/=total_images
    NCI_suc/=total_images
    Avg_IOU/=total_images
                
    table = PrettyTable()
    table.field_names = ["dataset", "NCI", "NFI", "NFO", "Avg_IOU", "#samples", "#clicks"]
    table.add_row([dataset_name, round(NCI_all,2), NFI, NFO, Avg_IOU, total_num_instances, max_interactions])

    print(table)