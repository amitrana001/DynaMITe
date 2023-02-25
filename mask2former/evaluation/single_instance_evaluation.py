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
from mask2former.evaluation.eval_utils import post_process, compute_iou, get_next_click, prepare_scribbles

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

    # model_name = cfg.MODEL.WEIGHTS.split("/")[-2] + f"_S{sampling_strategy}"
    # save_vis_path = os.path.join("./output/evaluation", dataset_name, f"{model_name}_S{sampling_strategy}_{start_time}/")
    
    # save_stats_path = os.path.join("./output/evaluation",  f'{dataset_name}.txt')
    # if not os.path.exists(save_stats_path):
    #     header = ["model","NOC_80", "NOC_85", "NOC_90", "NFO_80","NFO_85","NFO_90","IOU_80","IOU_85", "IOU_90","#samples","#clicks"]
    #     with open(save_stats_path, 'w') as f:
    #         writer = csv.writer(f, delimiter= "\t")
    #         writer.writerow(header)

    # import pickle
    # from collections import defaultdict
    # pt_sampled_dict = defaultdict(list)
    # with open("output/pt_sampled_dict.pickle", 'rb') as f:
    #     pt_sampled_dict = pickle.load(f)

    # features_dicts = {}
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
                coords = inputs[0]["coords"]
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
            batched_num_scrbs_per_mask) = model(inputs)
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
            # if num_interactions >= max_interactions:
            #     # print(inputs[0]["image_id"])
            #     for iou in ious:
            #         if iou<iou_threshold:
            #             num_failed_objects+=1
            # for iou in ious:
            #     total_iou += iou

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

    # now = datetime.datetime.now()
    # # dd/mm/YY H:M:S
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S_")
    # features_dict_path = f"output/{cfg.DATASETS.TEST[0]}_features_dict_{dt_string}.pickle"
    # points_dict_path = f"output/{cfg.DATASETS.TEST[0]}_points_dict_{dt_string}.pickle"
    # with open(features_dict_path, 'wb') as handle:
    #     pickle.dump(features_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open(points_dict_path, 'wb') as handle:
    #             pickle.dump(pt_sampled_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
    # NOC_80, NFO_80, IOU_80 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.80)
    # NOC_85, NFO_85, IOU_85 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.85)
    # NOC_90, NFO_90, IOU_90 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.90)
    #
    # row = [model_name, NOC_80, NOC_85, NOC_90, NFO_80, NFO_85, NFO_90, IOU_80, IOU_85, IOU_90, total_num_instances, max_interactions]
    # with open(save_stats_path, 'a') as f:
    #     writer = csv.writer(f, delimiter= "\t")
    #     writer.writerow(row)
    #
    # from prettytable import PrettyTable
    # table = PrettyTable()
    # table.field_names = ["dataset","NOC_80", "NOC_85", "NOC_90", "NFO_80","NFO_85","NFO_90","#samples", "#clicks"]
    # table.add_row([dataset_name, NOC_80, NOC_85, NOC_90, NFO_80, NFO_85, NFO_90, total_num_instances, max_interactions])
    #
    # print(table)
    # with EventStorage() as s:
    # if comm.is_main_process():
    #     # storage = get_event_storage()
    
    #     storage = get_event_storage()
    #     storage.put_scalar(f"NOC_{iou_threshold*100}", total_num_interactions/total_num_instances)
    #     storage.put_scalar("Avg IOU", total_iou/total_num_instances)
    #     storage.put_scalar("Failed Cases", num_failed_objects)
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


def get_summary(dataset_iou_list, max_clicks=20, iou_thres=0.85):

    num_images =len(dataset_iou_list)
    total_clicks = 0
    failed_objects = 0
    total_iou = 0
    for (key,per_image_iou_list) in dataset_iou_list.items():
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
