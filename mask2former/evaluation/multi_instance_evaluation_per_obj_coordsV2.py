# Copyright (c) Facebook, Inc. and its affiliates.
import csv
import datetime
import logging
import os
import time
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
import torchvision
from collections import defaultdict

from detectron2.utils.colormap import colormap
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from torch import nn

from mask2former.evaluation.eval_utils import prepare_scribbles, get_next_coords_bg_eval, get_next_coords_fg_eval, save_visualization
from mask2former.utils.train_sampling_utils import compute_fn_iou
color_map = colormap(rgb=True, maximum=1)


def evaluate(
    model, data_loader, cfg, dataset_name=None, save_stats_summary =True, 
    iou_threshold = 0.85, max_interactions = 10, sampling_strategy=0
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.
    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        max_interactions: int
            Maxinum number of interactions per object
        iou_threshold: float
            IOU threshold bwteen gt_mask and pred_mask to stop interaction
    Returns:
        The return value of `evaluator.evaluate()`
    """
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
   
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0 
    total_compute_time = 0
    total_eval_time = 0

    save_results_path = os.path.join("./output/evaluations/", dataset_name)
    # save_results_path += cfg.DATASETS.TEST[0]

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        use_prev_logits = False
        # variables to get evaluation summary statistics
        total_num_instances = 0
        total_num_interactions = 0
        num_failed_objects=0
        total_iou = 0.0
        failed_images_ids = []
        total_failed_images = 0
        avg_num_clicks_per_images = [] #total clikcs for image / total instances in image
        avg_over_total_images = 0
        failed_objects_areas = [0] * 201 #binning the object area ratio
        bin_size = 200
        time_per_image_features = []
        time_per_intreaction_tranformer_decoder = []
        time_per_image_annotation = []

        clicked_objects_per_interaction = defaultdict(list)
        ious_objects_per_interaction = defaultdict(list)
        num_instances_per_image = {}

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            if 'bg_mask' not in inputs[0]:
                continue
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            
            gt_masks = inputs[0]['instances'].gt_masks.to('cpu')
            # bg_mask = inputs[0]["bg_mask"].to('cpu')
            semantic_map = inputs[0]['semantic_map'].to('cpu')
            # comb_gt_fg_mask = torch.max(gt_masks,dim=0).values

            num_instances, h_t, w_t = gt_masks.shape[:]
            total_num_instances+=num_instances

            num_instances_per_image[f"{inputs[0]['image_id']}_{idx}"] = num_instances
            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = num_instances
            # stop_interaction = False
            ious = [0.0]*num_instances
            num_clicks_per_object = [1]*(num_instances+1) # 1 for background
            num_clicks_per_object[-1] = 0

            radius = 3
            max_iters_for_image = max_interactions * num_instances
            not_clicked_map = np.ones_like(gt_masks[0], dtype=np.bool)

            if sampling_strategy == 0:
                # coords = inputs[0]["coords"]
                for coords_list in inputs[0]['fg_click_coords']:
                    for coords in coords_list:
                        not_clicked_map[coords[0], coords[1]] = False
            elif sampling_strategy == 1:
                all_scribbles = torch.cat(inputs[0]['fg_scrbs']).to('cpu')
                point_mask = torch.max(all_scribbles,dim=0).values
                not_clicked_map[torch.where(point_mask)] = False
            fg_click_map = bg_click_map = None

            start_features_time = time.perf_counter()
            (processed_results, outputs, images, scribbles,
            num_insts, features, mask_features,
            transformer_encoder_features, multi_scale_features,
            batched_num_scrbs_per_mask,batched_fg_coords_list,
            batched_bg_coords_list) = model(inputs)

            orig_device = images.tensor.device
            time_per_image_features.append(time.perf_counter() - start_features_time)
            time_per_image = time.perf_counter() - start_features_time
            # save_visualization(inputs[0], gt_masks, scribbles[0], save_results_path,  ious[0], num_interactions-1,  alpha_blend=0.6)
            pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
            pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)
            
            ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold)
            # save_visualization(inputs[0], pred_masks, scribbles[0], save_results_path,  ious[0], num_interactions,  alpha_blend=0.6)
            
            point_sampled = True

            clicked_objects_per_interaction[f"{inputs[0]['image_id']}_{idx}"].append([True]*(num_instances+1))
            ious_objects_per_interaction[f"{inputs[0]['image_id']}_{idx}"].append(ious)
            while (num_interactions<max_iters_for_image and point_sampled):
                if all(iou >= iou_threshold for iou in ious):
                    break

                index_clicked = [False]*(num_instances+1)
                indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=False).indices
                point_sampled = False
                for i in indexes:
                    if ious[i]<iou_threshold and num_clicks_per_object[i] != max_interactions:
                        # total_num_interactions+=1
                        (scrbs, is_fg, obj_index, not_clicked_map, coords,
                        fg_click_map, bg_click_map) = get_next_click(pred_masks[i], gt_masks[i], semantic_map, not_clicked_map, fg_click_map,
                                                                    bg_click_map, orig_device, radius, sampling_strategy, padding=True,
                                                                    )
                        total_num_interactions+=1
                        scrbs = prepare_scribbles(scrbs,images)
                        if obj_index == -1:
                            if batched_bg_coords_list[0]:
                                scribbles[0][-1] = torch.cat((scribbles[0][-1],scrbs))
                                batched_bg_coords_list[0].extend([[coords[0], coords[1],num_interactions]])
                            else:
                                scribbles[0][-1] = scrbs
                                batched_bg_coords_list[0] = [[coords[0], coords[1],num_interactions]]
                            num_clicks_per_object[-1]+=1
                            point_sampled = True
                            index_clicked[-1] = True
                        else:
                            scribbles[0][obj_index] = torch.cat([scribbles[0][obj_index], scrbs], 0)
                            batched_num_scrbs_per_mask[0][obj_index] += 1
                            batched_fg_coords_list[0][obj_index].extend([[coords[0], coords[1],num_interactions]])
                        
                            num_clicks_per_object[i]+=1
                            index_clicked[obj_index] = True
                        point_sampled = True
                        break
                if point_sampled:
                    num_interactions+=1
                    clicked_objects_per_interaction[f"{inputs[0]['image_id']}_{idx}"].append(index_clicked)
                    prev_mask_logits=None    

                    start_transformer_decoder_time = time.perf_counter()           
                    (processed_results, outputs, images, scribbles,
                    num_insts, features, mask_features, transformer_encoder_features,
                    multi_scale_features, batched_num_scrbs_per_mask, batched_fg_coords_list,
                    batched_bg_coords_list)= model(inputs, images, scribbles, num_insts,
                                                features, mask_features, transformer_encoder_features,
                                                multi_scale_features, prev_mask_logits,
                                                batched_num_scrbs_per_mask,
                                                batched_fg_coords_list, batched_bg_coords_list)
                    time_per_intreaction_tranformer_decoder.append(time.perf_counter() - start_transformer_decoder_time)
                    time_per_image+=time.perf_counter() - start_transformer_decoder_time


                    pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
                    pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)
                    
                    ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold)
                    # save_visualization(inputs[0], pred_masks, scribbles[0], save_results_path,  ious[0], num_interactions,  alpha_blend=0.6)
            
                    ious_objects_per_interaction[f"{inputs[0]['image_id']}_{idx}"].append(ious)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_per_image_annotation.append(time_per_image)
            avg_num_clicks_per_images.append(num_interactions/num_instances)
            
            # if num_interactions >= max_iters_for_image:
            if any(iou < iou_threshold for iou in ious):
                # print(inputs[0]["image_id"])
                failed_images_ids.append(inputs[0]["image_id"])
                for i, iou in enumerate(ious):
                    if iou<iou_threshold:
                        # indx = gt_masks[i].sum()//bin_size
                        indx = gt_masks[i].sum()/(h_t * w_t)
                        indx = int(indx*bin_size)
                        if indx < len(failed_objects_areas):
                            failed_objects_areas[indx] +=1
                        else:
                            failed_objects_areas[-1] += 1
                        
                        num_failed_objects+=1
            for iou in ious:
                total_iou += iou

            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            # eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        # f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"Total instances: {total_num_instances}. "
                        f"Average interactions:{(total_num_interactions/total_num_instances):.2f}. "
                        f"Avg IOU: {(total_iou/total_num_instances):.3f} "
                        f"Failed Instances: {num_failed_objects} "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        ),
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = {'total_num_instances': [total_num_instances],
               'total_num_interactions': [total_num_interactions],
               'num_failed_objects': [num_failed_objects],
               'total_iou': [total_iou],
               'failed_images_ids': failed_images_ids,
               'failed_objects_areas': failed_objects_areas,
               'avg_num_clicks_per_images': avg_num_clicks_per_images,
               'total_compute_time_str': total_compute_time_str,
               'iou_threshold': iou_threshold,
               'time_per_intreaction_tranformer_decoder': time_per_intreaction_tranformer_decoder,
               'time_per_image_features': time_per_image_features,
               'time_per_image_annotation': time_per_image_annotation,
               'clicked_objects_per_interaction': [clicked_objects_per_interaction],
               'ious_objects_per_interaction': [ious_objects_per_interaction],
               'num_instances_per_image': [num_instances_per_image],
               }
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def compute_iou(gt_masks, pred_masks, ious, iou_threshold):
    for i in range(len(ious)):
        intersection = (gt_masks[i] * pred_masks[i]).sum()
        union = torch.logical_or(gt_masks[i], pred_masks[i]).to(torch.int).sum()
        ious[i] = intersection/union
    return ious

def compute_fn_iou_eval(gt_masks, pred_masks, bg_mask, max_objs=15, iou_thres = 0.90):

    fn_ratio = np.zeros(gt_masks.shape[0])
    for i, (gt_mask, pred_mask) in enumerate(zip(gt_masks,pred_masks)):
        # _pred_mask = np.logical_and(pred_mask,gt_mask)
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
        fn_area = fn.sum()
        gt_area = gt_mask.sum()
        fn_ratio[i] = fn_area/gt_area

    indices = torch.topk(torch.tensor(fn_ratio), len(fn_ratio),largest=True).indices
    
    return indices

from mask2former.data.points.annotation_generator import create_circular_mask, get_max_dt_point_mask
def get_next_click(pred_mask, gt_mask, semantic_map, not_clicked_map, fg_click_map,
                   bg_click_map, device, radius, sampling_strategy, padding=True,
):
    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)

    fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)
    
    # if fn_mask.sum()==0:
    #     fn_mask = gt_mask
    
    H, W = gt_mask.shape

    if padding:
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    import cv2
    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    if padding:
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]


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

    obj_index = semantic_map[coords_y[0]][coords_x[0]] - 1
    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    
    if sampling_strategy == 0:
        not_clicked_map[coords_y[0], coords_x[0]] = False
    elif sampling_strategy == 1:
        not_clicked_map[np.where(pm==1)] = False
   
    return (torch.from_numpy(pm).to(device, dtype = torch.float).unsqueeze(0),
            is_positive, obj_index, not_clicked_map, sample_locations[0],
            fg_click_map, bg_click_map)



    