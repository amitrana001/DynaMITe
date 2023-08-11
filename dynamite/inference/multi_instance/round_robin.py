# Copyright (c) Facebook, Inc. and its affiliates.
import csv
import datetime
import logging
import os
import time
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
import random
import torchvision
from collections import defaultdict

from detectron2.utils.colormap import colormap
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from torch import nn
from ..utils.clicker import Clicker
from ..utils.predictor import Predictor

def evaluate(
    model, data_loader, iou_threshold = 0.85, max_interactions = 10, sampling_strategy=1,
    eval_strategy = "round_robin", seed_id = 0, vis_path = None
):
    """
    Run model on the data_loader and return a dict, later used to calculate
    all the metrics for multi-instance inteactive segmentation such as NCI,
    NFO, NFI, and Avg IoU.
    The model will be used in eval mode.

    Arguments:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        iou_threshold: float
            Desired IoU value for each object mask
        max_interactions: int
            Maxinum number of interactions per object
        sampling_strategy: int
            Strategy to avaoid regions while sampling next clicks
            0: new click sampling avoids all the previously sampled click locations
            1: new click sampling avoids all locations upto radius 5 around all
               the previously sampled click locations
        eval_strategy: str
            Click sampling strategy during refinement
        seed_id: int
            Used to generate fixed seed during evaluation
        vis_path: str
            Path to save visualization of masks with clicks during evaluation

    Returns:
        Dict with following keys:
            'total_num_instances': total number of instances in the dataset
            'total_num_interactions': total number of interactions/clicks sampled 
            'total_compute_time_str': total compute time for evaluating the dataset
            'iou_threshold': iou_threshold
            'num_interactions_per_image': a dict with keys as image ids and values 
             as total number of interactions per image
            'final_iou_per_object': a dict with keys as image ids and values as
             list of ious of all objects after final interaction
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
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        # variables to get evaluation summary statistics
        total_num_instances = 0
        total_num_interactions = 0
        
        final_iou_per_object = defaultdict(list)
        num_interactions_per_image = {}
                
        random.seed(123456+seed_id)
        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            
            clicker = Clicker(inputs, sampling_strategy)
            predictor = Predictor(model)

            if vis_path:
                clicker.save_visualization(vis_path, ious=[0], num_interactions=0)
            num_instances = clicker.num_instances
            total_num_instances+=num_instances

            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = num_instances
            num_clicks_per_object = [1]*(num_instances+1) # 1 for background
            num_clicks_per_object[-1] = 0

            max_iters_for_image = max_interactions * num_instances
        
            pred_masks = predictor.get_prediction(clicker)
            clicker.set_pred_masks(pred_masks)
            ious = clicker.compute_iou()

            if vis_path:
                clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions)

            point_sampled = True

            random_indexes = list(range(len(ious)))

            final_iou_per_object[f"{inputs[0]['image_id']}_{idx}"].append(ious)

            curr_sel_obj = -1
            cont_clicks_per_obj = [1]*num_instances
            max_cont_clicks = 10

            while (num_interactions<max_iters_for_image and point_sampled):
                if all(iou >= iou_threshold for iou in ious):
                    break

                index_clicked = [False]*(num_instances+1)
                
                if all((iou>=iou_threshold or c >= max_cont_clicks) for iou,c in zip(ious,cont_clicks_per_obj)):
                    cont_clicks_per_obj = [0]*num_instances
                    curr_sel_obj = -1
                    
                point_sampled = False
                if (curr_sel_obj != -1 and cont_clicks_per_obj[curr_sel_obj] < max_cont_clicks
                    and ious[curr_sel_obj] < iou_threshold):
                    obj_index = clicker.get_next_click(refine_obj_index=curr_sel_obj, time_step=num_interactions)
                    total_num_interactions+=1
                    
                    index_clicked[obj_index] = True
                    num_clicks_per_object[curr_sel_obj]+=1
                    point_sampled = True
                    cont_clicks_per_obj[curr_sel_obj]+=1
                else:
                    random.shuffle(random_indexes)
                    indexes = random_indexes
                    for i in indexes:
                        if ious[i]<iou_threshold and cont_clicks_per_obj[i] < max_cont_clicks: 
                            obj_index = clicker.get_next_click(refine_obj_index=i, time_step=num_interactions)
                            total_num_interactions+=1
                            
                            index_clicked[obj_index] = True
                            num_clicks_per_object[i]+=1
                            point_sampled = True
                            curr_sel_obj = i
                            # to handle first selected object, since we start with one click per object
                            temp = cont_clicks_per_obj[curr_sel_obj]+1
                            cont_clicks_per_obj = [0]*num_instances
                            cont_clicks_per_obj[curr_sel_obj] = temp
                            break
                if point_sampled:
                    num_interactions+=1
                    
                    pred_masks = predictor.get_prediction(clicker)
                    clicker.set_pred_masks(pred_masks)
                    ious = clicker.compute_iou()
                    
                    if vis_path:
                        clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            final_iou_per_object[f"{inputs[0]['image_id']}_{idx}"].append(ious)
            num_interactions_per_image[f"{inputs[0]['image_id']}_{idx}"] = num_interactions

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
                        # f"Avg IOU: {(total_iou/total_num_instances):.3f} "
                        # f"Failed Instances: {num_failed_objects} "
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
                'total_compute_time_str': total_compute_time_str,
                'iou_threshold': iou_threshold,
                'final_iou_per_object': [final_iou_per_object],
                'num_interactions_per_image': [num_interactions_per_image],
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

