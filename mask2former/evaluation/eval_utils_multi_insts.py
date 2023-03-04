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
from mask2former.data.points.annotation_generator import create_circular_mask, get_max_dt_point_mask
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

def compute_iou(gt_masks, pred_masks, ious, iou_threshold, ignore_masks=None):

    if ignore_masks is None:
        for i in range(len(ious)):
            intersection = (gt_masks[i] * pred_masks[i]).sum()
            union = torch.logical_or(gt_masks[i], pred_masks[i]).to(torch.int).sum()
            # if ious[i] < iou_threshold:
            #     ious[i]= intersection/union
            # else:
            #     ious[i]= max(intersection/union, ious[i])
            ious[i] = intersection/union
            # print(ious)
        return ious

    for i in range(len(ious)):
        # intersection = (gt_masks[i] * pred_masks[i]).sum()
        # union = torch.logical_or(gt_masks[i], pred_masks[i]).to(torch.int).sum()
        n_iou = get_iou_per_mask(gt_masks[i], pred_masks[i], ignore_masks[i])
        ious[i] = n_iou
        # if ious[i] < iou_threshold:
        #     ious[i]= n_iou
        # else:
        #     ious[i]= max(n_iou, ious[i])
    # print(ious)
    return ious

def get_iou_per_mask(gt_mask, pred_mask, ignore_mask):
    # ignore_gt_mask_inv = gt_mask != ignore_label
    ignore_gt_mask_inv = ~(ignore_mask.to(dtype=torch.bool))
    # ignore_gt_mask_inv = 
    obj_gt_mask = gt_mask

    intersection = torch.logical_and(torch.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = torch.logical_and(torch.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union

def save_visualization(inputs, pred_masks, scribbles, dir_path, iou, num_iter,  alpha_blend=0.3):
    
    image = np.asarray(inputs['image'].permute(1,2,0))

    visualizer = Visualizer(image, metadata=None)
    pred_masks = F.resize(pred_masks.to(dtype=torch.uint8), image.shape[:2])
    c = []
    for i in range(pred_masks.shape[0]):
        # c.append(color_map[2*(i)+2]/255.0)
        c.append(color_map[i]/255.0)
    # pred_masks = np.asarray(pred_masks).astype(np.bool_)
    vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c, alpha=alpha_blend)
    # [Optional] prepare labels

    image = vis.get_image()
    # # Laminate your image!
    total_colors = len(color_map)-1
    
    h,w = image.shape[:2]
    for i, scrb in enumerate(scribbles[:-1]):
        scrb = torch.max(scrb,0).values.to('cpu')
        scrb = scrb[:h, :w]
        color = np.array(color_map[total_colors-5*i-4], dtype=np.uint8)
        image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
        
    if scribbles[-1] is not None:
        scrb = torch.max(scribbles[-1],0).values.to('cpu')
        scrb = scrb[:h, :w]
        color = np.array([255,0,0], dtype=np.uint8)
        image[scrb>0.5, :] = np.array(color, dtype=np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (inputs["width"],inputs["height"]))
    save_dir = os.path.join(dir_path, str(inputs['image_id']))
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"iter_{num_iter}_{np.round(iou,4)}.jpg"), image)

def get_gt_clicks_coords_eval(masks, max_num_points=1, ignore_masks=None, radius_size=8, first_click_center=True, t= 0, unique_timestamp=False):

    """
    :param masks: numpy array of shape I x H x W
    :param patch_size: size of patch (int)
    """
    # assert all_masks is not None
    masks = np.asarray(masks).astype(np.uint8)
    if ignore_masks is not None:
        not_ignores_mask = np.logical_not(np.asarray(ignore_masks, dtype=np.bool_))

    I, H, W = masks.shape
    num_scrbs_per_mask = [0]*I
    fg_coords_list = []
    fg_point_masks = []
    for i, (_m) in enumerate(masks):
        coords = []
        point_masks_per_obj = []
        if first_click_center:
            if ignore_masks is not None:
                _m = np.logical_and(_m, not_ignores_mask[i]).astype(np.uint8)
            center_coords = get_max_dt_point_mask(_m, max_num_pts=max_num_points)
            # center_coords.append(t)
            _pm = create_circular_mask(H, W, centers=[center_coords], radius=radius_size)
            point_masks_per_obj.append(_pm)
            coords.append([center_coords[0], center_coords[1], t])
            if unique_timestamp:
                t+=1
            num_scrbs_per_mask[i]+=1 
        fg_coords_list.append(coords)
        fg_point_masks.append(torch.from_numpy(np.stack(point_masks_per_obj, axis=0)).to(torch.uint8))
    return num_scrbs_per_mask, fg_coords_list, None, fg_point_masks, None

def get_next_coords_bg_eval(all_fp, device, not_clicked_map ,fg_click_map, bg_click_map, radius = 3, strategy = 0):

    H,W = all_fp.shape
    fp_mask = all_fp
    if strategy == 2:
        fp_mask = np.logical_and(fp_mask, ~(bg_click_map))
    
    fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]

    fp_mask_dt = fp_mask_dt * not_clicked_map
    _max_dist = np.max(fp_mask_dt)
    coords_y, coords_x = np.where(fp_mask_dt == _max_dist)

    sample_locations = [[coords_y[0], coords_x[0]]]
    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    
    if strategy == 0:
        not_clicked_map[coords_y[0], coords_x[0]] = False
    elif strategy == 1:
        not_clicked_map[np.where(pm==1)] = False
    elif strategy == 2:
        bg_click_map = np.logical_or(bg_click_map,pm)

    return (torch.from_numpy(pm).to(device, dtype = torch.float).unsqueeze(0),
            not_clicked_map, sample_locations[0],
            fg_click_map, bg_click_map)

def get_next_coords_fg_eval(pred_mask, gt_mask, not_clicked_map, fg_click_map, bg_click_map, device, radius=3, strategy=0):

    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    
    if fn_mask.sum()==0:
        fn_mask = gt_mask

    H, W = gt_mask.shape    
    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    fn_mask_dt = fn_mask_dt * not_clicked_map

    fn_max_dist = np.max(fn_mask_dt)
    coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
    sample_locations = [[coords_y[0], coords_x[0]]]
    # print(sample_locations)
    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
   
    if strategy == 0:
        not_clicked_map[coords_y[0], coords_x[0]] = False
    elif strategy == 1:
        not_clicked_map[np.where(pm==1)] = False
    elif strategy == 2:
        fg_click_map = np.logical_or(bg_click_map,pm)

    return (torch.from_numpy(pm).to(device, dtype = torch.float).unsqueeze(0),
            not_clicked_map, sample_locations[0],
            fg_click_map, bg_click_map)

def get_next_click_bg(all_fp, not_clicked_map, radius=5, num_points=1,device='cpu'):
    H, W = all_fp.shape
    # all_fp = np.asarray(all_fp).astype(np.uint8)
    all_fp = np.asarray(all_fp, dtype = np.bool_)
    # print("all_fp:", all_fp.sum())
    # pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    # fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    # # fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask)
    # H, W = gt_mask.shape
    padding=True
    if padding:
        all_fp = np.pad(all_fp, ((1, 1), (1, 1)), 'constant')
        # fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    all_fp_dt = cv2.distanceTransform(all_fp.astype(np.uint8), cv2.DIST_L2, 0)
    # fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    if padding:
        all_fp_dt = all_fp_dt[1:-1, 1:-1]
        # fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    all_fp_dt = all_fp_dt * not_clicked_map
    # fp_mask_dt = fp_mask_dt * not_clicked_map

    all_fp_max_dt = np.max(all_fp_dt)
    # fp_max_dist = np.max(fp_mask_dt)

    # is_positive = fn_max_dist > fp_max_dist
    # if is_positive:
    coords_y, coords_x = np.where(all_fp_dt == all_fp_max_dt)  # coords is [y, x]
    # else:
    #     coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]
    sample_locations = [[coords_y[0], coords_x[0]]]
    # print(sample_locations)
    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    not_clicked_map[coords_y[0], coords_x[0]] = False
    return torch.from_numpy(pm).to(device, dtype = torch.uint8), not_clicked_map

def get_fn_area(pred_masks, gt_masks):
    fn_per_object = []
    for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks,gt_masks)):
        pred_mask = pred_mask>0.5
        fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).sum()
        fn_per_object.append(fn)
    return fn_per_object

def get_next_click_fg(pred_mask, gt_mask, not_clicked_map, radius=5, device='cpu'):

    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    
    if fn_mask.sum()==0:
        fn_mask = gt_mask

    H, W = gt_mask.shape
    
    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    fn_mask_dt = fn_mask_dt * not_clicked_map

    fn_max_dist = np.max(fn_mask_dt)
    coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
    # else:
    #     coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]
    sample_locations = [[coords_y[0], coords_x[0]]]
    # print(sample_locations)
    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    not_clicked_map[coords_y[0], coords_x[0]] = False
    return torch.from_numpy(pm).to(device, dtype = torch.uint8), not_clicked_map

def get_next_click(
    pred_mask, gt_mask, not_clicked_map, radius=8, device='cpu',
    ignore_mask=None, padding=True, strategy = 1,
    fg_click_map = None, bg_click_map = None
):

    if ignore_mask is not None:
        not_ignore_mask = np.logical_not(np.asarray(ignore_mask, dtype=np.bool_))
    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)

    if ignore_mask is not None:
        fn_mask =  np.logical_and(np.logical_and(gt_mask, np.logical_not(pred_mask)), not_ignore_mask)
        fp_mask =  np.logical_and(np.logical_and(np.logical_not(gt_mask), pred_mask), not_ignore_mask)
    else:
        fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
        fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)
    
    if strategy == 2:
        fn_mask = np.logical_and(fn_mask,~(fg_click_map))
        fp_mask = np.logical_and(fp_mask, ~(bg_click_map))
    
    if fn_mask.sum()==0:
        fn_mask = gt_mask
    
    H, W = gt_mask.shape

    if padding:
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    if padding:
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    if strategy !=2:
        fn_mask_dt = fn_mask_dt * not_clicked_map
        fp_mask_dt = fp_mask_dt * not_clicked_map

    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    is_positive = fn_max_dist > fp_max_dist

    if is_positive:
        coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
    else:
        coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

    sample_locations = [[coords_y[0], coords_x[0]]]

    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    
    if strategy == 0:
        not_clicked_map[coords_y[0], coords_x[0]] = False
    elif strategy == 1:
        not_clicked_map[np.where(pm==1)] = False
    elif strategy == 2:
        if is_positive:
            fg_click_map = np.logical_or(fg_click_map,pm)
        else:
            bg_click_map = np.logical_or(bg_click_map,pm)

    return (torch.from_numpy(pm).to(device, dtype = torch.float).unsqueeze(0),
            is_positive, not_clicked_map, sample_locations[0],
            fg_click_map, bg_click_map)

def prepare_scribbles(scribbles,images):
    h_pad, w_pad = images.tensor.shape[-2:]
    padded_scribbles = torch.zeros((scribbles.shape[0],h_pad, w_pad), dtype=scribbles.dtype, device=scribbles.device)
    padded_scribbles[:, : scribbles.shape[1], : scribbles.shape[2]] = scribbles
    return padded_scribbles


def log_single_instance(res, max_interactions, dataset_name, model_name):
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

    NOC_80, NFO_80, IOU_80 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.80)
    NOC_85, NFO_85, IOU_85 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.85)
    NOC_90, NFO_90, IOU_90 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.90)

    row = [model_name,
           NOC_80, NOC_85, NOC_90, NFO_80, NFO_85, NFO_90, IOU_80, IOU_85, IOU_90,
           total_num_instances,
           max_interactions]

    save_stats_path = os.path.join("./output/evaluation/final_eval", f'{dataset_name}.txt')
    os.makedirs("./output/evaluation/final_eval", exist_ok=True)
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


def log_multi_instance(res, max_interactions, dataset_name, model_name, iou_threshold=0.85, save_stats_summary=True,
                       per_obj = True, sampling_strategy = 0):
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

    # header = ['Model Name', 'IOU_thres', 'Avg_NOC', 'NOF', "Avg_IOU", "max_num_iters", "num_inst"]
    NOC = np.round(total_num_interactions / total_num_instances, 2)
    NCI = sum(res['avg_num_clicks_per_images']) / len(res['avg_num_clicks_per_images'])
    NFI = len(res['failed_images_ids'])
    Avg_IOU = np.round(total_iou / total_num_instances, 4)
    row = [model_name, NCI, NFI, NOC, num_failed_objects, Avg_IOU, iou_threshold, max_interactions, total_num_instances]

    # save_stats_path = os.path.join("./output/evaluation", f'{dataset_name}.txt')
    save_stats_path = os.path.join("./output/evaluation/",  f'{dataset_name}.txt')
    if not os.path.exists(save_stats_path):
        # print("No File")
        header = ['Model Name', 'NCI', 'NFI','NOC', 'NFO', "Avg_IOU", 'IOU_thres',"max_num_iters", "num_inst"]
        with open(save_stats_path, 'w') as f:
            writer = csv.writer(f, delimiter= "\t")
            writer.writerow(header)
            # writer.writerow(row)

    with open(save_stats_path, 'a') as f:
        writer = csv.writer(f, delimiter= "\t")
        writer.writerow(row)

    if save_stats_summary:
        summary_stats = {}
        summary_stats["sampling_strategy"] = f"S_{sampling_strategy}"
        summary_stats["dataset"] = dataset_name
        summary_stats["model"] = model_name
        summary_stats["iou_threshold"] = iou_threshold
        summary_stats["failed_images_counts"] = NFI
        summary_stats["avg_over_total_images"] = NCI
        summary_stats["Avg_NOC"] = NOC
        summary_stats["Avg_IOU"] = np.round(total_iou / total_num_instances, 4)
        summary_stats["num_failed_objects"] = num_failed_objects
        summary_stats["failed_images_ids"] = res['failed_images_ids']
        summary_stats["failed_objects_areas"] = res['failed_objects_areas']
        summary_stats["avg_num_clicks_per_images"] = np.mean(res['avg_num_clicks_per_images'])
        summary_stats["total_computer_time"] = res['total_compute_time_str']
        summary_stats["time_per_intreaction_tranformer_decoder"] = np.mean(
            res['time_per_intreaction_tranformer_decoder']
        )
        summary_stats["time_per_image_features"] = np.mean(res['time_per_image_features'])
        summary_stats["time_per_image_annotation"] = np.mean(res['time_per_image_annotation'])
        summary_stats["clicked_objects_per_interaction"] = res["clicked_objects_per_interaction"]
        summary_stats["ious_objects_per_interaction"] = res["ious_objects_per_interaction"]
        summary_stats["ious_objects_per_interaction"] = res["ious_objects_per_interaction"]
        summary_stats['num_instances_per_image'] =  res['num_instances_per_image']

        now = datetime.datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S_")
        if per_obj:
            model_name += "_per_obj_"
        model_name += dt_string
        save_summary_path = os.path.join(f"./output/evaluations/{dataset_name}")
        os.makedirs(save_summary_path, exist_ok=True)
        stats_file = os.path.join(save_summary_path,
                                  f"{model_name}_{max_interactions}.pickle")

        with open(stats_file, 'wb') as handle:
            pickle.dump(summary_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_summary(dataset_iou_list, max_clicks=20, iou_thres=0.85):

    num_images =len(dataset_iou_list)
    total_clicks = 0
    failed_objects = 0
    total_iou = 0
    for (key, per_image_iou_list) in dataset_iou_list.items():
        vals = per_image_iou_list>=iou_thres
        if np.any(vals):
            num_clicks =  np.argmax(vals) + 1
            total_iou += per_image_iou_list[num_clicks-1]
        else:
            num_clicks =  max_clicks
            total_iou += per_image_iou_list[-1]
            failed_objects+=1
        total_clicks+=num_clicks
    
    return np.round(total_clicks/num_images,2), failed_objects, np.round(total_iou/num_images,4)