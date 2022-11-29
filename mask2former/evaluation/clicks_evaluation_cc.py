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
from ..data.scribble.gen_scribble import get_iterative_scribbles, get_iterative_eval
from ..utils.iterative_misc import preprocess_batch_data, get_new_scribbles
from mask2former.data.points.annotation_generator import get_corrective_points, get_next_click, get_corrective_points_determinstic
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

    save_results_path = os.path.join("./all_data/evaluations/", cfg.DATASETS.TEST[0], "swin_small/")
    
    # total_iou = 0.0
    save_evaluation_path = os.path.join("./all_data/evaluations/",  f'{cfg.DATASETS.TEST[0]}.txt')
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
            
            processed_results, outputs, images, _, _, features, mask_features, transformer_encoder_features, multi_scale_features = model(inputs)
            # outputs = model(inputs)
            # images = None
            scribbles = None
            num_insts = None
            # ### Interaction loop
            orig_device = inputs[0]['instances'].gt_masks.device
            
            gt_masks = inputs[0]['instances'].gt_masks.to('cpu')
            pred_masks = processed_results[0]['instances'].pred_masks.to('cpu')
            bg_mask = inputs[0]["bg_mask"].to('cpu')

            num_instances, h_t, w_t = gt_masks.shape[:]
            h,w = pred_masks.shape[1:]
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
            # breakpoint()
            while (num_interactions<max_interactions):
                
                # if num_interactions>=2:
                #     radius=5
                # TO DO
                # don't change the masks with iou 80%
                pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
                pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)
                # print(torch.unique(pred_masks))
                for i in range(pred_masks.shape[0]):
                    pred_masks[i] = torch.logical_or(pred_masks[i], inputs[0]['fg_scrbs'][i]).to(dtype=torch.uint8)

                if is_post_process:
                    pred_masks = post_process(pred_masks,inputs[0]['fg_scrbs'],ious,iou_threshold)
                
                if ignore_masks is None:
                    ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold)
                else:
                    ious = compute_iou_new(gt_masks,pred_masks,ignore_masks,ious,iou_threshold)
                # break
                # save_image(inputs, pred_masks,save_results_path, ious[0], num_interactions, alpha_blend=0.3)

                if all(iou >= iou_threshold for iou in ious):
                    # stop_interaction=True
                    break
                else:
                    new_scrbs = []
                    # gt_masks = torchvision.transforms.Resize(size = (h,w))(gt_masks)
                    all_fp = torch.logical_and(bg_mask, torch.max(pred_masks,dim=0).values)
                    
                    flag = False
                    for i,(gt_mask, pred_mask) in enumerate(zip(gt_masks, pred_masks)):
                        if ious[i] < iou_threshold:
                            # flag=True
                            # total_num_interactions+=1
                            # scrbs, is_fg = get_iterative_scribbles(pred_mask, gt_mask, bg_mask, device=orig_device)
                            scrbs, is_fg = get_corrective_points_determinstic(pred_mask, gt_mask, bg_mask, device=orig_device,fg_points=inputs[0]['fg_scrbs'][i],radius=radius)
                            new_scrbs.append(scrbs)

                            total_num_interactions+=1
                            if is_fg:
                                # total_num_interactions+=1
                                fg = torchvision.transforms.Resize(size = (h_t, w_t))(scrbs[0].unsqueeze(0)).squeeze(0)
                                inputs[0]['fg_scrbs'][i] = torch.logical_or(inputs[0]['fg_scrbs'][i], fg)
                            else:
                                # continue
                                bg = torchvision.transforms.Resize(size = (h_t, w_t))(scrbs[0].unsqueeze(0))
                                bg_mask = torch.logical_and(bg_mask, ~(bg.squeeze(0).to(dtype=torch.bool)))
                                if inputs[0]['bg_scrbs'] is None:
                                    inputs[0]['bg_scrbs'] = bg
                                else:
                                    inputs[0]['bg_scrbs'] = torch.cat((inputs[0]['bg_scrbs'],bg))
                    if flag:
                        bg_scrbs = generate_point_all_fp(all_fp,radius=5,num_points=1,device=orig_device)
                        total_num_interactions+=1
                        if inputs[0]['bg_scrbs'] is None:
                            inputs[0]['bg_scrbs'] = bg_scrbs
                        else:
                            inputs[0]['bg_scrbs'] = torch.cat((inputs[0]['bg_scrbs'],bg_scrbs))
                        bg_mask = torch.logical_and(bg_mask, ~(bg_scrbs.squeeze(0).to(dtype=torch.bool)))  
                if use_prev_logits:
                    processed_results, outputs, _, _, _, features, mask_features, transformer_encoder_features, multi_scale_features= model(inputs, images, scribbles, num_insts,
                                                                        features, mask_features, transformer_encoder_features,
                                                                        multi_scale_features, outputs['pred_masks'])
                else:
                    images = None
                    processed_results, outputs, images, _, _, features, mask_features, transformer_encoder_features, multi_scale_features= model(inputs, images, scribbles, num_insts,
                                                                        features, mask_features, transformer_encoder_features,
                                                                        multi_scale_features)
                
                num_interactions+=1
                # break
            # break
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

            # start_eval_time = time.perf_counter()
            # evaluator.process(inputs, outputs)
            # total_eval_time += time.perf_counter() - start_eval_time

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

def compute_iou(gt_masks, pred_masks, ious, iou_threshold):
    for i in range(len(ious)):
        intersection = (gt_masks[i] * pred_masks[i]).sum()
        union = torch.logical_or(gt_masks[i], pred_masks[i]).to(torch.int).sum()
        if ious[i] < iou_threshold:
            ious[i]= intersection/union
        else:
            ious[i]= max(intersection/union, ious[i])
    # print(ious)
    return ious

def compute_iou_new(gt_masks, pred_masks, ignore_masks, ious, iou_threshold):
    for i in range(len(ious)):
        # intersection = (gt_masks[i] * pred_masks[i]).sum()
        # union = torch.logical_or(gt_masks[i], pred_masks[i]).to(torch.int).sum()
        n_iou = get_iou(gt_masks[i], pred_masks[i],ignore_masks[i])
        if ious[i] < iou_threshold:
            ious[i]= n_iou
        else:
            ious[i]= max(n_iou, ious[i])
    # print(ious)
    return ious

def get_iou(gt_mask, pred_mask, ignore_mask):
    # ignore_gt_mask_inv = gt_mask != ignore_label
    ignore_gt_mask_inv = ~(ignore_mask.to(dtype=torch.bool))
    # ignore_gt_mask_inv = 
    obj_gt_mask = gt_mask

    intersection = torch.logical_and(torch.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = torch.logical_and(torch.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union

def save_image(inputs, pred_masks, dir_path, iou, num_iter, alpha_blend=0.3):

    image = np.asarray(inputs[0]['image'].permute(1,2,0))
    # h,w = inputs[0]['height'], inputs[0]['width']
    # image = cv2.resize(image, (h,w))
    image = image*(1-alpha_blend)
    # pred_masks = torchvision.transforms.Resize(size = (h,w))(pred_masks)

    for i in range(pred_masks.shape[0]):
        # (mask.detach().cpu().numpy() * 255).astype(np.uint8)
        mask = np.asarray(pred_masks[i]).astype(np.uint8)
        color_mask = np.zeros_like(image, dtype=np.uint8)
        # color_mask[:,:,1] = 1
        color_mask[:,:,:] = np.array(color_map[i+1]*255, dtype=np.uint8)
        if len(mask.shape) == 2:
            mask = mask[:,:,None]
        # image = (image*(1-alpha_blend) + color_mask*mask*alpha_blend).astype(np.uint8)
        image = (image + color_mask*mask*alpha_blend).astype(np.uint8)
        # image = cv2.addWeighted(image[:,:,::-1], 1, (color_mask*mask)[:,:,::-1], alpha_blend,0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    total_colors = len(color_map)-1
    if inputs[0]['fg_scrbs'] is not None:
        for i, scrb in enumerate(inputs[0]['fg_scrbs']):
            # color = [np.random.randint(0, 255), np.random.randint(0, 1), np.random.randint(0, 255)]
            color = np.array(color_map[total_colors-20-i]*255, dtype=np.uint8)
            image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
    if inputs[0]['bg_scrbs'] is not None:
        # print(total_colors, inputs[0]['bg_scrbs'].shape)
        for i, scrb in enumerate(inputs[0]['bg_scrbs']):
            # color = [np.random.randint(0, 255), np.random.randint(0, 1), np.random.randint(0, 255)]
            color = np.array(color_map[total_colors-1]*255, dtype=np.uint8)
            image[scrb>0.5, :] = np.array(color, dtype=np.uint8)

    img_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    save_dir = os.path.join(dir_path, str(inputs[0]['image_id']))
    # print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"iter_{num_iter}_{iou}.jpg"), img_write)
    # return image

def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def post_process(pred_masks,scribbles,ious=None, iou_threshold = 0.85):
    out = []
    # print(pred_masks.shape)
    # print(scribbles.shape)
    for (pred_mask, points, iou) in zip(pred_masks,scribbles, ious):
        if iou < iou_threshold:
            # opening_size = np.random.randint(5, 20)
            # pred_mask = cv2.morphologyEx(np.asarray(pred_mask), cv2.MORPH_OPEN, disk_kernel(opening_size))
            num_labels, labels_im = cv2.connectedComponents(np.asarray(pred_mask).astype(np.uint8))
            points_comp = labels_im[torch.where(points==1)]
            # print(points_comp)
            vals,counts = np.unique(points_comp, return_counts=True)
            cc_mask = np.zeros_like(labels_im)
            for val in vals:
                cc_mask = np.logical_or(cc_mask, labels_im==val)
            # index = np.argmax(counts)
            # print(vals[index])
            # pred_mask = torch.from_numpy(labels_im==vals[index])

            # pred_mask = pred_mask.to(dtype=torch.uint8)
            pred_mask = torch.from_numpy(cc_mask).to(dtype=torch.uint8)
        out.append(pred_mask)


    return torch.stack(out,0)

from mask2former.data.points.annotation_generator import point_candidates_dt_determinstic, create_circular_mask
def generate_point_all_fp(all_fp, radius=5, num_points=1,device='cpu'):
    H, W = all_fp.shape
    all_fp = np.asarray(all_fp).astype(np.uint8)
    sample_locations = point_candidates_dt_determinstic(np.asarray(all_fp), max_num_pts=num_points)
    pm = np.zeros_like(all_fp)
    scribbles = []
    if sample_locations.shape[0] > 0:
        pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    scribbles.append(torch.from_numpy(pm).to(device,dtype = torch.uint8))
    return torch.stack(scribbles,0)
