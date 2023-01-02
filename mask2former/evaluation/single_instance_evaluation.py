# Copyright (c) Facebook, Inc. and its affiliates.
from audioop import mul
import datetime
import logging
import time
import os
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from traceback import walk_tb
from typing import List, Union
import torch
import torchvision
from torch import nn
import cv2
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.structures import BitMasks
import csv
import numpy as np
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from mask2former.data.points.annotation_generator import get_next_click
from mask2former.evaluation.eval_utils import post_process, compute_iou, get_next_click, save_visualization, prepare_scribbles
from detectron2.utils.colormap import colormap
color_map = colormap(rgb=True, maximum=1)

def get_avg_noc(
    model, data_loader, cfg, evaluator=None,iou_threshold = 0.85,
    max_interactions = 20, is_post_process = False
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

    save_results_path = os.path.join("./output/", cfg.DATASETS.TEST[0], "swin_small/")
    
    # total_iou = 0.0
    save_evaluation_path = os.path.join("./output/",  f'{cfg.DATASETS.TEST[0]}.txt')
    if not os.path.exists(save_evaluation_path):
        # print("No File")
        header = ['Model Name', 'IOU_thres', 'Avg_NOC', 'NOF', "Avg_IOU", "max_num_iters", "num_inst"]
        with open(save_evaluation_path, 'w') as f:
            writer = csv.writer(f, delimiter= "\t")
            writer.writerow(header)

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
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            
            # orig_device = inputs[0]['instances'].gt_masks.device
            
            gt_masks = inputs[0]['instances'].gt_masks.to('cpu')
            bg_mask = inputs[0]["bg_mask"].to('cpu')
            not_clicked_map = np.ones_like(gt_masks[0], dtype=np.bool)
            
            num_instances, h_t, w_t = gt_masks.shape[:]
            total_num_instances+=num_instances
            
            ignore_masks = None
            if 'ignore_mask' in inputs[0]:
                ignore_masks = inputs[0]['ignore_mask'].to(device='cpu', dtype = torch.uint8)
                ignore_masks =  torchvision.transforms.Resize(size = (h_t,w_t))(ignore_masks)
                # ignore_masks = ignore_masks>128
            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = 1
            # stop_interaction = False
            ious = [0.0]*num_instances
            radius = 8
            
            (processed_results, outputs, images, scribbles,
            num_insts, features, mask_features,
            transformer_encoder_features, multi_scale_features,
            batched_num_scrbs_per_mask) = model(inputs)
            orig_device = images.tensor.device

            # save_visualization(inputs[0], gt_masks, scribbles[0], save_results_path,  ious[0], num_interactions-1,  alpha_blend=0.6)
            pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
            pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)

            if is_post_process:
                pred_masks = post_process(pred_masks,inputs[0]['fg_scrbs'],ious,iou_threshold)
            
            ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold,ignore_masks)
            # save_visualization(inputs[0], pred_masks, scribbles[0], save_results_path,  ious[0], num_interactions,  alpha_blend=0.6)
            
            while (num_interactions<max_interactions):
                
                if all(iou >= iou_threshold for iou in ious):
                    break
                # don't change the masks with iou 80%
                for i,(gt_mask, pred_mask) in enumerate(zip(gt_masks, pred_masks)):
                    if ious[i] < iou_threshold:
                        scrbs, is_fg, not_clicked_map= get_next_click(pred_mask, gt_mask, not_clicked_map,
                                                                     radius=radius, device=orig_device,
                                                                     ignore_mask=ignore_masks[0])

                        total_num_interactions+=1
                        scrbs = prepare_scribbles(scrbs,images)
                        if is_fg:
                            scribbles[0][i] = torch.cat([scribbles[0][i], scrbs], 0)
                            batched_num_scrbs_per_mask[0][i] += 1
                        else:
                            if scribbles[0][-1] is None:
                                scribbles[0][-1] = scrbs
                            else:
                                scribbles[0][-1] = torch.cat((scribbles[0][-1],scrbs))
                
                (processed_results, outputs, images, scribbles,
                num_insts, features, mask_features, transformer_encoder_features,
                multi_scale_features, batched_num_scrbs_per_mask)= model(inputs, images, scribbles, num_insts,
                                                                        features, mask_features, transformer_encoder_features,
                                                                        multi_scale_features, batched_num_scrbs_per_mask=batched_num_scrbs_per_mask)
                
                pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
                pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)

                if is_post_process:
                    pred_masks = post_process(pred_masks,inputs[0]['fg_scrbs'],ious,iou_threshold)
                
                ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold,ignore_masks)
                num_interactions+=1
                # save_visualization(inputs[0], pred_masks, scribbles[0], save_results_path,  ious[0], num_interactions,  alpha_blend=0.6)
                
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if num_interactions >= max_interactions:
                # print(inputs[0]["image_id"])
                for iou in ious:
                    if iou<iou_threshold:
                        num_failed_objects+=1
            for iou in ious:
                total_iou += iou

            ###--------------
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
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

    logger.info(
        "Total number of instances: {}, Average num of interactions:{}".format(
            total_num_instances, total_num_interactions/total_num_instances
        )
    )
    logger.info(
        "Total number of failed cases: {}, Avg IOU: {}".format(
            num_failed_objects, total_iou/total_num_instances
        ) 
    )
    # header = ['Model Name', 'IOU_thres', 'Avg_NOC', 'NOF', "Avg_IOU", "max_num_iters", "num_inst"]
    model_name = cfg.MODEL.WEIGHTS.split("/")[-2]
    if is_post_process:
        model_name+="_p"
    Avg_NOC = np.round(total_num_interactions/total_num_instances,2)
    Avg_IOU = np.round(total_iou/total_num_instances, 2)

    row = [model_name, iou_threshold, Avg_NOC, num_failed_objects, Avg_IOU, max_interactions, total_num_instances]
    with open(save_evaluation_path, 'a') as f:
        writer = csv.writer(f, delimiter= "\t")
        writer.writerow(row)
    
    # with EventStorage() as s:
    if comm.is_main_process():
        # storage = get_event_storage()
    
        storage = get_event_storage()
        storage.put_scalar(f"NOC_{iou_threshold*100}", total_num_interactions/total_num_instances)
        storage.put_scalar("Avg IOU", total_iou/total_num_instances)
        storage.put_scalar("Failed Cases", num_failed_objects)
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    results = None
    if results is None:
        results = {}
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
