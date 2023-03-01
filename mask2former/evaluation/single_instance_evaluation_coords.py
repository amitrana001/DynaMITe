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

from mask2former.data.points.annotation_generator import get_next_click
from mask2former.evaluation.eval_utils import compute_iou, get_next_click, prepare_scribbles

color_map = colormap(rgb=True, maximum=1)

def get_avg_noc(
    model, data_loader, cfg, iou_threshold = 0.85, dataset_name= None,
    max_interactions = 20, is_post_process = False, sampling_strategy=2
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
            gt_masks = inputs[0]['instances'].gt_masks.to('cpu')
            bg_mask = inputs[0]["bg_mask"].to('cpu')
            
            not_clicked_map = np.ones_like(gt_masks[0], dtype=np.bool)
            if sampling_strategy == 0:
                # coords = inputs[0]["coords"]
                coords = inputs[0]['fg_click_coords'][0][0][:2]
                not_clicked_map[coords[0], coords[1]] = False
            elif sampling_strategy == 1:
                point_mask = inputs[0]['fg_scrbs'][0][0].to('cpu')
                not_clicked_map[torch.where(point_mask)] = False
            # elif sampling_strategy == 2:
            fg_click_map = np.asarray(inputs[0]['fg_scrbs'][0][0].to('cpu'),dtype=np.bool_)
            bg_click_map = np.zeros_like(fg_click_map,dtype=np.bool_)

            num_instances, h_t, w_t = gt_masks.shape[:]
            total_num_instances+=num_instances
            
            ignore_masks = None
            if 'ignore_mask' in inputs[0]:
                ignore_masks = inputs[0]['ignore_mask'].to(device='cpu', dtype = torch.uint8)
                ignore_masks =  torchvision.transforms.Resize(size = (h_t,w_t))(ignore_masks)

            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = 1
            ious = [0.0]*num_instances
            radius = 8
            
            (processed_results, outputs, images, scribbles,
            num_insts, features, mask_features,
            transformer_encoder_features, multi_scale_features,
            batched_num_scrbs_per_mask,batched_fg_coords_list,
            batched_bg_coords_list) = model(inputs)
            orig_device = images.tensor.device

            # save_visualization(inputs[0], gt_masks, scribbles[0], save_vis_path,  ious[0], num_interactions-1,  alpha_blend=0.6)
            pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
            pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)

            ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold,ignore_masks)
            # save_visualization(inputs[0], pred_masks, scribbles[0], save_vis_path,  ious[0], num_interactions,  alpha_blend=0.6)
            per_image_iou_list.append(ious[0].item())
            while (num_interactions<max_interactions):
                
                if all(iou >= iou_threshold for iou in ious):
                    break
                # if num_interactions>3:
                #     radius=5
                for i,(gt_mask, pred_mask) in enumerate(zip(gt_masks, pred_masks)):
                    if ious[i] < iou_threshold:
                        scrbs, is_fg, not_clicked_map, coords, fg_click_map, bg_click_map = get_next_click(pred_mask, gt_mask, not_clicked_map,
                                                                    radius=radius, device=orig_device,
                                                                    ignore_mask=ignore_masks[0] if ignore_masks!=None else None,
                                                                    strategy=sampling_strategy, fg_click_map = fg_click_map,
                                                                    bg_click_map = bg_click_map
                                                                )

                        # pt_sampled_dict[inputs[0]['image_id']].append(coords)
                        total_num_interactions+=1
                        scrbs = prepare_scribbles(scrbs,images)
                        if is_fg:
                            scribbles[0][i] = torch.cat([scribbles[0][i], scrbs], 0)
                            batched_num_scrbs_per_mask[0][i] += 1
                            # last_coords = batched_fg_coords_list[0][i][-1]
                            # prev_timestamp = last_coords[-1]
                            # new_timestamp = prev_timestamp +1
                            batched_fg_coords_list[0][i].extend([[coords[0], coords[1],num_interactions]])
                        else:
                            # to-do handling of new timestamp 
                            if batched_bg_coords_list[0]:
                                scribbles[0][-1] = torch.cat((scribbles[0][-1],scrbs))
                                batched_bg_coords_list[0].extend([[coords[0], coords[1],num_interactions]])
                            else:
                                scribbles[0][-1] = scrbs
                                batched_bg_coords_list[0] = [[coords[0], coords[1],num_interactions]]
                prev_mask_logits=None               
                (processed_results, outputs, images, scribbles,
                num_insts, features, mask_features, transformer_encoder_features,
                multi_scale_features, batched_num_scrbs_per_mask, batched_fg_coords_list,
                batched_bg_coords_list)= model(inputs, images, scribbles, num_insts,
                                               features, mask_features, transformer_encoder_features,
                                               multi_scale_features, prev_mask_logits,
                                               batched_num_scrbs_per_mask,
                                               batched_fg_coords_list, batched_bg_coords_list)
                
                pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
                pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)
                
                ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold,ignore_masks)
                per_image_iou_list.append(ious[0].item())
                num_interactions+=1
                # save_visualization(inputs[0], pred_masks, scribbles[0], save_vis_path,  ious[0], num_interactions,  alpha_blend=0.6)
            
            dataset_iou_list[inputs[0]['image_id']] = np.asarray(per_image_iou_list)
            
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
