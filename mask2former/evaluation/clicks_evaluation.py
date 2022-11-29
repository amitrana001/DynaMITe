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

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage

import numpy as np
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from ..data.scribble.gen_scribble import get_iterative_scribbles, get_iterative_eval
from ..utils.iterative_misc import preprocess_batch_data, get_new_scribbles
from mask2former.data.points.annotation_generator import get_corrective_points, get_next_click, get_corrective_points_determinstic
from detectron2.utils.colormap import colormap
color_map = colormap(rgb=True, maximum=1)

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:
        .. code-block:: python
            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.
    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def get_avg_noc(
    model, data_loader, cfg, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    use_clicks=True, use_prev_mask = False,
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
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    max_interactions =  cfg.ITERATIVE.TEST.MAX_NUM_INTERACTIONS - 1
    iou_threshold = cfg.ITERATIVE.TEST.IOU_THRESHOLD
    # use_prev_logits = use_prev_mask
    # print(os.getcwd())
    save_results_path = os.path.join("./all_data/evaluations/", cfg.DATASETS.TEST[0], "argmax_th85_v01/")
    # save_results_path += cfg.DATASETS.TEST[0]
    # print(save_results_path)

    # use_prev_logits = False
    # # total number of object instances
    # total_num_instances = 0
    # total_num_interactions = 0
    # num_failed_objects=0
    # total_iou = 0.0
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
            # print(gt_masks.shape)
            # print(outputs['pred_masks'].shape)
            # print(pred_masks.shape)

            # we start with atleast one interaction per instance
            total_num_interactions+=(num_instances)

            num_interactions = 0
            # stop_interaction = False
            ious = [0.0]*num_instances
            
            while (num_interactions<max_interactions):
                
                # TO DO
                # don't change the masks with iou 80%
                pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
                pred_masks = torchvision.transforms.Resize(size = (h_t,w_t))(pred_masks)
                # from torch.nn import functional as F

                
                ious = compute_iou(gt_masks,pred_masks,ious,iou_threshold)
                # break
                # save_image(inputs, pred_masks,save_results_path, ious[0], num_interactions, alpha_blend=0.3)
                if all(iou >= iou_threshold for iou in ious):
                    # stop_interaction=True
                    break
                else:
                    new_scrbs = []
                    # gt_masks = torchvision.transforms.Resize(size = (h,w))(gt_masks)
                    for i,(gt_mask, pred_mask) in enumerate(zip(gt_masks, pred_masks)):
                        if ious[i] < iou_threshold:
                            # total_num_interactions+=1
                            # scrbs, is_fg = get_iterative_scribbles(pred_mask, gt_mask, bg_mask, device=orig_device)
                            scrbs, is_fg = get_corrective_points_determinstic(pred_mask, gt_mask, bg_mask, device=orig_device)
                            new_scrbs.append(scrbs)

                            total_num_interactions+=1
                            if is_fg:
                                fg = torchvision.transforms.Resize(size = (h_t, w_t))(scrbs[0].unsqueeze(0)).squeeze(0)
                                inputs[0]['fg_scrbs'][i] = torch.logical_or(inputs[0]['fg_scrbs'][i], fg)
                            else:
                                bg = torchvision.transforms.Resize(size = (h_t, w_t))(scrbs[0].unsqueeze(0))
                                if inputs[0]['bg_scrbs'] is None:
                                    inputs[0]['bg_scrbs'] = bg
                                else:
                                    inputs[0]['bg_scrbs'] = torch.cat((inputs[0]['bg_scrbs'],bg))     
                if use_prev_logits:
                    processed_results, outputs, _, _, _, features, mask_features, transformer_encoder_features, multi_scale_features = model(inputs, images, scribbles, num_insts,
                                                                        features, mask_features, transformer_encoder_features,
                                                                        multi_scale_features, outputs['pred_masks'])
                else:
                    images = None
                    processed_results, outputs, images, _, _, features, mask_features, transformer_encoder_features, multi_scale_features = model(inputs, images, scribbles, num_insts,
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

    if comm.is_main_process():
        storage = get_event_storage()
        storage.put_scalar(f"NOC_{iou_threshold*100}", total_num_interactions/total_num_instances)
        storage.put_scalar("Avg IOU", total_iou/total_num_instances)
        storage.put_scalar("Failed Cases", num_failed_objects)

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
    # results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
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
            ious[i]= max(torch.div(intersection/union), ious[i])
    # print(ious)
    return ious
import cv2

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
        for i, scrb in enumerate(inputs[0]['bg_scrbs']):
            # color = [np.random.randint(0, 255), np.random.randint(0, 1), np.random.randint(0, 255)]
            color = np.array(color_map[total_colors-i]*255, dtype=np.uint8)
            image[scrb>0.5, :] = np.array(color, dtype=np.uint8)

    img_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    save_dir = os.path.join(dir_path, str(inputs[0]['image_id']))
    # print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"iter_{num_iter}_{iou}.jpg"), img_write)
    # return image

def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def post_process(pred_masks,scribbles):
    out = []
    # print(pred_masks.shape)
    # print(scribbles.shape)
    for (pred_mask, points) in zip(pred_masks,scribbles):
        # opening_size = np.random.randint(5, 20)
        # pred_mask = cv2.morphologyEx(np.asarray(pred_mask), cv2.MORPH_OPEN, disk_kernel(opening_size))
        num_labels, labels_im = cv2.connectedComponents(np.asarray(pred_mask))
        points_comp = labels_im[torch.where(points==1)]
        # print(points_comp)
        vals,counts = np.unique(points_comp, return_counts=True)
        index = np.argmax(counts)
        # print(vals[index])
        pred_mask = torch.from_numpy(labels_im==vals[index])
        pred_mask = pred_mask.to(dtype=torch.uint8)
        
        out.append(pred_mask)

    return torch.stack(out,0)