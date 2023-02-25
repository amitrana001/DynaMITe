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
from detectron2.utils.colormap import colormap
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from torch import nn

from mask2former.evaluation.eval_utils import prepare_scribbles, get_fn_area, get_next_click_bg, get_next_click_fg

color_map = colormap(rgb=True, maximum=1)


def evaluate(
    model, data_loader, cfg, dataset_name=None, save_stats_summary =True, 
    iou_threshold = 0.85, max_interactions = 10
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

    save_results_path = os.path.join("./output/evaluation/", dataset_name, "swin_base_/")
    # save_results_path += cfg.DATASETS.TEST[0]
   
    save_stats_path = os.path.join("./output/evaluation/",  f'{dataset_name}.txt')
    if not os.path.exists(save_stats_path):
        # print("No File")
        header = ['Model Name', 'NCI', 'NFI','NOC', 'NFO', "Avg_IOU", 'IOU_thres',"max_num_iters", "num_inst"]
        with open(save_stats_path, 'w') as f:
            writer = csv.writer(f, delimiter= "\t")
            writer.writerow(header)

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
            bg_mask = inputs[0]["bg_mask"].to('cpu')
            comb_gt_fg_mask = torch.max(gt_masks,dim=0).values

            num_instances, h_t, w_t = gt_masks.shape[:]
            total_num_instances+=num_instances
            
            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = num_instances
            # stop_interaction = False
            ious = [0.0]*num_instances
            num_clicks_per_object = [1]*(num_instances+1) # 1 for background
            num_clicks_per_object[-1] = 0

            radius = 5
            max_iters_for_image = max_interactions * num_instances
            not_clicked_map = np.ones_like(gt_masks[0], dtype=np.bool)

            start_features_time = time.perf_counter()
            (processed_results, outputs, images, scribbles,
            num_insts, features, mask_features,
            transformer_encoder_features, multi_scale_features,
            batched_num_scrbs_per_mask) = model(inputs)
            orig_device = images.tensor.device
            time_per_image_features.append(time.perf_counter() - start_features_time)
            time_per_image = time.perf_counter() - start_features_time
            # save_visualization(inputs[0], gt_masks, scribbles[0], save_results_path,  ious[0], num_interactions-1,  alpha_blend=0.6)
            pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
            pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)
            
            ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold)
            # save_visualization(inputs[0], pred_masks, scribbles[0], save_results_path,  ious[0], num_interactions,  alpha_blend=0.6)
            
            point_sampled = True
            num_times_point_smapled_false = 0
            time_transformer_decoder_loop = 0.0

            while (num_interactions<max_iters_for_image and (any(c < max_interactions for c in num_clicks_per_object))):
                # if num_interactions>=2:
                #     radius=3
                if all(iou >= iou_threshold for iou in ious) or num_times_point_smapled_false >= 2:
                    break
                
                comb_pred_fg_mask = torch.max(pred_masks,dim=0).values.to(dtype=torch.bool)
                comb_fp = torch.logical_and(bg_mask, comb_pred_fg_mask).to(dtype=torch.uint8)
                comb_fn = torch.logical_and(torch.logical_not(comb_pred_fg_mask), comb_gt_fg_mask).to(dtype=torch.uint8)

                fn_area_per_object = get_fn_area(pred_masks,gt_masks)
                
                if (comb_fp.sum() >= comb_fn.sum() or (not point_sampled)) and (num_clicks_per_object[-1]!=max_interactions):
                    bg_scrbs, not_clicked_map = get_next_click_bg(comb_fp,not_clicked_map, radius=radius,num_points=1,device=orig_device)
                    total_num_interactions+=1
                    # print(torch.all(bg_scrbs==0))
                    scrbs = prepare_scribbles(bg_scrbs.unsqueeze(0),images)
                    if scribbles[0][-1] is None:
                        scribbles[0][-1] = scrbs
                    else:
                        scribbles[0][-1] = torch.cat((scribbles[0][-1],scrbs))
                    num_clicks_per_object[-1]+=1
                    point_sampled = True
                else:
                    worst = len(fn_area_per_object)
                    indxs = torch.topk(torch.tensor(fn_area_per_object), min(worst, len(fn_area_per_object)),largest=True).indices
                    at_least_one_fg = False
                    for i in indxs:
                        if (num_clicks_per_object[i]!=max_interactions) and ious[i]<iou_threshold:
                            total_num_interactions+=1
                            scrbs, not_clicked_map= get_next_click_fg(pred_masks[i], gt_masks[i],not_clicked_map)
                            
                            scrbs = prepare_scribbles(scrbs.unsqueeze(0),images)
                            scribbles[0][i] = torch.cat([scribbles[0][i], scrbs.to(device=orig_device)], 0)
                            batched_num_scrbs_per_mask[0][i] += 1
                            num_clicks_per_object[i]+=1
                            at_least_one_fg = True
                            break
                    point_sampled = at_least_one_fg
                if point_sampled:
                    num_interactions+=1
                    start_transformer_decoder_time = time.perf_counter()
                    (processed_results, outputs, images, scribbles,
                    num_insts, features, mask_features,
                    transformer_encoder_features, multi_scale_features,
                    batched_num_scrbs_per_mask) = model(inputs, images, scribbles, num_insts,
                                                        features, mask_features, transformer_encoder_features,
                                                        multi_scale_features, batched_num_scrbs_per_mask=batched_num_scrbs_per_mask)
                    
                    pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
                    pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)
                    
                    ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold)
                    time_per_intreaction_tranformer_decoder.append(time.perf_counter() - start_transformer_decoder_time)
                    time_per_image+=time.perf_counter() - start_transformer_decoder_time
                if not point_sampled:
                    num_times_point_smapled_false+=1
                else:
                    num_times_point_smapled_false-=1
            
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

    # logger.info(
    #     "Total number of instances: {}, Average num of interactions:{}".format(
    #         total_num_instances, total_num_interactions/total_num_instances
    #     )
    # )
    # logger.info(
    #     "Total number of failed cases: {}, Avg IOU: {}".format(
    #         num_failed_objects, total_iou/total_num_instances
    #     )
    # )
    #
    # # header = ['Model Name', 'IOU_thres', 'Avg_NOC', 'NOF', "Avg_IOU", "max_num_iters", "num_inst"]
    # model_name = cfg.MODEL.WEIGHTS.split("/")[-2]
    # NOC = np.round(total_num_interactions/total_num_instances,2)
    # NCI = sum(avg_num_clicks_per_images)/len(avg_num_clicks_per_images)
    # NFI = len(failed_images_ids)
    # Avg_IOU = np.round(total_iou/total_num_instances, 4)
    # ['Model Name', 'NCI', 'NFI','NOC', 'NFO', "Avg_IOU", 'IOU_thres',"max_num_iters", "num_inst"]
    # row = [model_name, NCI, NFI, NOC, num_failed_objects, Avg_IOU, iou_threshold, max_interactions, total_num_instances]
    # with open(save_stats_path, 'a') as f:
    #     writer = csv.writer(f, delimiter= "\t")
    #     writer.writerow(row)
    #
    # if save_stats_summary:
    #     summary_stats = {}
    #     summary_stats["dataset"] = dataset_name
    #     summary_stats["model"] = model_name
    #     summary_stats["iou_threshold"] = iou_threshold
    #     summary_stats["failed_images_counts"] = NFI
    #     summary_stats["avg_over_total_images"] = NCI
    #     summary_stats["Avg_NOC"] = NOC
    #     summary_stats["Avg_IOU"] = np.round(total_iou/total_num_instances, 4)
    #     summary_stats["num_failed_objects"] = num_failed_objects
    #     summary_stats["failed_images_ids"] = failed_images_ids
    #     summary_stats["failed_objects_areas"] = failed_objects_areas
    #     summary_stats["avg_num_clicks_per_images"] = avg_num_clicks_per_images
    #     summary_stats["total_computer_time"] = total_compute_time_str
    #     summary_stats["time_per_intreaction_tranformer_decoder"] = time_per_intreaction_tranformer_decoder
    #     summary_stats["time_per_image_features"] = time_per_image_features
    #     summary_stats["time_per_image_annotation"] = time_per_image_annotation
    #
    #     save_summary_path = os.path.join("./all_data/evaluations/", cfg.DATASETS.TEST[0])
    #     stats_file = os.path.join(save_summary_path, f"{model_name}_{max_interactions}_max_click_th_final_updated_time_summary.pickle")
    #
    #     import pickle
    #     with open(stats_file, 'wb') as handle:
    #         pickle.dump(summary_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # results = None
    # if results is None:
    #     results = {}
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

    