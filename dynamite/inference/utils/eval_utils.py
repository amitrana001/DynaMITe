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
from dynamite.data.points.annotation_generator import create_circular_mask, get_max_dt_point_mask
from offline_summary import get_statistics

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


def log_single_instance(res, max_interactions, dataset_name, model_name, ablation=False, save_summary_stats=False):
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
    
    if save_summary_stats:
        save_summary_path = os.path.join(f"./output/evaluation/final/summary/{dataset_name}")
        os.makedirs(save_summary_path, exist_ok=True)
        stats_file = os.path.join(save_summary_path,
                                    f"{model_name}_{max_interactions}.pickle")

        with open(stats_file, 'wb') as handle:
            pickle.dump(dataset_iou_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    NOC_80, NFO_80, IOU_80 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.80)
    NOC_85, NFO_85, IOU_85 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.85)
    NOC_90, NFO_90, IOU_90 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.90)

    row = [model_name,
           NOC_80, NOC_85, NOC_90, NFO_80, NFO_85, NFO_90, IOU_80, IOU_85, IOU_90,
           total_num_instances,
           max_interactions]

    if ablation:
        save_stats_path = os.path.join("./output/evaluation/ablation", f'{dataset_name}.txt')
        os.makedirs("./output/evaluation/ablation", exist_ok=True)
    else:
        save_stats_path = os.path.join("./output/evaluation/final", f'{dataset_name}.txt')
        os.makedirs("./output/evaluation/final", exist_ok=True)

    if not os.path.exists(save_stats_path):
        header = ["model",
                  "NOC_80", "NOC_85", "NOC_90", "NFO_80","NFO_85","NFO_90","IOU_80","IOU_85", "IOU_90",
                  "#samples","#clicks"]
        with open(save_stats_path, 'w') as f:
            writer = csv.writer(f, delimiter= "\t")
            writer.writerow(header)
            # writer.writerow(row)
    
    with open(save_stats_path, 'a') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)

    
    table = PrettyTable()
    table.field_names = ["dataset", "NOC_80", "NOC_85", "NOC_90", "NFO_80", "NFO_85", "NFO_90", "#samples",
                         "#clicks"]
    table.add_row(
        [dataset_name, NOC_80, NOC_85, NOC_90, NFO_80, NFO_85, NFO_90, total_num_instances, max_interactions])

    print(table)


def log_multi_instance(res, max_interactions, dataset_name, model_name, ablation=False,iou_threshold=0.85,
                        save_stats_summary=True, sampling_strategy = 1):
    logger = logging.getLogger(__name__)
    total_num_instances = sum(res['total_num_instances'])
    total_num_interactions = sum(res['total_num_interactions'])
   
    logger.info(
        "Total number of instances: {}, Average num of interactions:{}".format(
            total_num_instances, total_num_interactions / total_num_instances
        )
    )

    if save_stats_summary:
        summary_stats = {}
        summary_stats["sampling_strategy"] = f"S_{sampling_strategy}"
        summary_stats["dataset"] = dataset_name
        summary_stats["model"] = model_name
        summary_stats["iou_threshold"] = iou_threshold
       
        summary_stats["total_computer_time"] = res['total_compute_time_str']
        summary_stats["time_per_intreaction_tranformer_decoder"] = np.mean(
            res['time_per_intreaction_tranformer_decoder']
        )
        summary_stats["time_per_image_features"] = np.mean(res['time_per_image_features'])
        summary_stats["time_per_image_annotation"] = np.mean(res['time_per_image_annotation'])

        ious_objects_per_interaction = {}
        for _d in res['ious_objects_per_interaction']:
            ious_objects_per_interaction.update(_d)
        
        click_sequence_per_image = {}
        for _d in res['click_sequence_per_image']:
            click_sequence_per_image.update(_d)
        
        object_areas_per_image = {}
        for _d in res['object_areas_per_image']:
            object_areas_per_image.update(_d)
        
        fg_click_coords_per_image = {}
        for _d in res['object_areas_per_image']:
            fg_click_coords_per_image.update(_d)
        
        bg_click_coords_per_image = {}
        for _d in res['object_areas_per_image']:
            bg_click_coords_per_image.update(_d)
       
        summary_stats["click_sequence_per_image"] = click_sequence_per_image
        summary_stats["ious_objects_per_interaction"] = ious_objects_per_interaction
        summary_stats['object_areas_per_image'] =  object_areas_per_image
        summary_stats['fg_click_coords_per_image'] =  fg_click_coords_per_image
        summary_stats['bg_click_coords_per_image'] =  bg_click_coords_per_image

        get_statistics(summary_stats)
        if ablation:
            save_summary_path = os.path.join(f"./output/evaluation/ablation/summary/{dataset_name}")
        else:
            save_summary_path = os.path.join(f"./output/evaluation/final/summary/{dataset_name}")
        os.makedirs(save_summary_path, exist_ok=True)
        stats_file = os.path.join(save_summary_path,
                                  f"{model_name}_{max_interactions}.pickle")
        
        with open(stats_file, 'wb') as handle:
            pickle.dump(summary_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        summary_stats = {}
        summary_stats["sampling_strategy"] = f"S_{sampling_strategy}"
        summary_stats["dataset"] = dataset_name
        summary_stats["model"] = model_name
        summary_stats["iou_threshold"] = iou_threshold
        ious_objects_per_interaction = {}
        for _d in res['ious_objects_per_interaction']:
            ious_objects_per_interaction.update(_d)
        summary_stats["ious_objects_per_interaction"] = ious_objects_per_interaction
        get_statistics(summary_stats)

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