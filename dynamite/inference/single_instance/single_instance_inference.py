# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
logging.basicConfig(level=logging.INFO)
import time
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
import torchvision
from detectron2.utils.colormap import colormap
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from torch import nn
from ..utils.clicker import Clicker
from ..utils.predictor import Predictor
color_map = colormap(rgb=True, maximum=1)

def get_avg_noc(
    model, data_loader, iou_threshold = 0.90, max_interactions = 20,
    sampling_strategy=1, vis_path = None
):
    """
    Run model on the data_loader and return a dict, later used to calculate
    all the metrics for single-instance inteactive segmentation such as NoC,
    NFO, and Avg IoU.
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
        vis_path: str
            Path to save visualization of masks with clicks during evaluation

    Returns:
        Dict with following keys:
            'total_num_instances': total number of instances in the dataset
            'total_num_interactions': total number of interactions/clicks sampled 
            'num_failed_objects': total number of failed objects
            'total_iou': sum of the ious of each object
            'dataset_iou_list': a dict with keys as image ids and values as
             list of ious of all objects after final interaction
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
            clicker = Clicker(inputs, sampling_strategy)
            predictor = Predictor(model)

            if vis_path:
                clicker.save_visualization(vis_path, ious=[0], num_interactions=0)

            num_instances = clicker.num_instances
            total_num_instances+=num_instances

            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = num_instances

            pred_masks = predictor.get_prediction(clicker)
            clicker.set_pred_masks(pred_masks)
            ious = clicker.compute_iou()

            if vis_path:
                clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions)

            per_image_iou_list.append(ious[0])
            while (num_interactions<max_interactions):
                
                if all(iou >= iou_threshold for iou in ious):
                    break

                if ious[0] < iou_threshold:
                    obj_index = clicker.get_next_click(refine_obj_index=0, time_step=num_interactions)
                    total_num_interactions+=1
                        
                num_interactions+=1        

                pred_masks = predictor.get_prediction(clicker)
                clicker.set_pred_masks(pred_masks)
                ious = clicker.compute_iou()

                if vis_path:
                    clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions)

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