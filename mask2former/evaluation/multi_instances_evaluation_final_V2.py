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

from .clicker import Clicker
color_map = colormap(rgb=True, maximum=1)


def evaluate(
    model, data_loader, cfg, dataset_name=None, save_stats_summary =True, 
    iou_threshold = 0.85, max_interactions = 10, sampling_strategy=0,
    eval_strategy = "worst", seed_id = 0, normalize_time = True
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

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        total_num_instances = 0
        total_num_interactions = 0
      
        time_per_image_annotation = []
        ious_objects_per_interaction = defaultdict(list)
        if eval_strategy == "random":
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
        
            predictor = Clicker(model, inputs, sampling_strategy, normalize_time=normalize_time)
            num_instances = predictor.num_instances
            total_num_instances+=num_instances

            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = num_instances
            num_clicks_per_object = [1]*(num_instances+1) # 1 for background
            num_clicks_per_object[-1] = 0

            max_iters_for_image = max_interactions * num_instances

            ious = predictor.predict()
        
            random_indexes = list(range(len(ious)))

            #interative refinement loop
            while (num_interactions<max_iters_for_image):
                if all(iou >= iou_threshold for iou in ious):
                    break

                if eval_strategy == "worst":
                    indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=False).indices
                elif eval_strategy == "best":
                    indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=True).indices
                elif eval_strategy == "random":
                    random.shuffle(random_indexes)
                    indexes = random_indexes
                else:
                    assert eval_strategy in ["worst", "best", "random"]

                # point_sampled = False
                for i in indexes:
                    if ious[i]<iou_threshold: 
                        obj_index = predictor.get_next_click(refine_obj_index=i, time_step=num_interactions)
                        num_clicks_per_object[i]+=1
                        # point_sampled = True
                        break
                # if point_sampled:
                total_num_interactions+=1
                num_interactions+=1
                ious = predictor.predict()
            
            ious_objects_per_interaction[f"{inputs[0]['image_id']}_{idx}"].append(ious)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"Total instances: {total_num_instances}. "
                        f"Average interactions:{(total_num_interactions/total_num_instances):.2f}. "
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
               'time_per_image_annotation': time_per_image_annotation,
               'ious_objects_per_interaction': [ious_objects_per_interaction],
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

    