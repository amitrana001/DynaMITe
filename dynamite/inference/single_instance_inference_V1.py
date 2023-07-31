# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
import torchvision
from detectron2.utils.colormap import colormap
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from torch import nn
from dynamite.evaluation.zoom_in import zoomIn
from dynamite.data.points.annotation_generator import get_next_click
from dynamite.evaluation.eval_utils import compute_iou, get_next_click, prepare_scribbles
from .clicker import Clicker
color_map = colormap(rgb=True, maximum=1)

def get_avg_noc(
    model, data_loader, cfg, iou_threshold = 0.85, dataset_name= None,
    max_interactions = 20, is_post_process = False, sampling_strategy=2,
    normalize_time= False
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
    logger.info("Using sampling strategy {}".format(sampling_strategy))

    total = len(data_loader)  # inference data loader must have a fixed length

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        # breakpoint()
        use_prev_logits = False
        # total number of object instances
        total_num_instances = 0
        total_num_interactions = 0
        num_failed_objects=0
        total_iou = 0.0
        start_data_time = time.perf_counter()
        dataset_iou_list = {}
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            
            per_image_iou_list = []
            predictor = Clicker(model, inputs, sampling_strategy, normalize_time=normalize_time)
            num_instances = predictor.num_instances
            total_num_instances+=num_instances

            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = num_instances
            
            start_features_time = time.perf_counter()
        
            ious = predictor.predict()

            # time_per_image_features.append(time.perf_counter() - start_features_time)
            time_per_image = time.perf_counter() - start_features_time
           
            # point_sampled = True
            per_image_iou_list.append(ious[0])
            while (num_interactions<max_interactions):
                
                if all(iou >= iou_threshold for iou in ious):
                    break
                # if num_interactions>3:
                #     radius=5
                # for i in range(len(ious)):
                if ious[0] < iou_threshold:
                    obj_index = predictor.get_next_click(refine_obj_index=0, time_step=num_interactions)
                    total_num_interactions+=1
                        
                num_interactions+=1

                start_transformer_decoder_time = time.perf_counter()           
                ious = predictor.predict()
                # time_per_intreaction_tranformer_decoder.append(time.perf_counter() - start_transformer_decoder_time)
                time_per_image+=time.perf_counter() - start_transformer_decoder_time
            
                    # ious_objects_per_interaction[f"{inputs[0]['image_id']}_{idx}"].append(ious)
                per_image_iou_list.append(ious[0])
               
            dataset_iou_list[f"{inputs[0]['image_id']}_{idx}"] = np.asarray(per_image_iou_list)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for iou in ious:
                total_iou += iou
                if iou<iou_threshold:
                    num_failed_objects+=1
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

    return {'total_num_instances': [total_num_instances],
            'total_num_interactions': [total_num_interactions],
            'num_failed_objects': [num_failed_objects],
            'total_iou': [total_iou],
            'dataset_iou_list': [dataset_iou_list]
            }

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

import cv2
from dynamite.data.points.annotation_generator import create_circular_mask, get_max_dt_point_mask
def get_next_click_V1(
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
    
    # if fn_mask.sum()==0:
    #     fn_mask = gt_mask
    
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
