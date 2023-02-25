# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import csv

import numpy as np

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from mask2former.evaluation.single_instance_evaluation import get_avg_noc, get_summary

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import (
    DatasetEvaluator,
)

# from mask2former.evaluation.iterative_evaluator import iterative_inference_on_dataset
# MaskFormer
from mask2former import (
    COCOLVISMultiInstMQCoordsDatasetMapper,
    DAVIS17DetmClicksDatasetMapper,
    COCOLVISMultiInstMQClicksDatasetMapper,
    COCOLVISSingleInstMQClicksDatasetMapper,
    COCOMultiInstStuffMultiQueriesClicksDatasetMapper,
    COCOSingleInstMultiQueriesStuffClicksDatasetMapper
)

from mask2former import (
    COCOMvalDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)


def log_single_instance(res, max_interactions, dataset_name, model_name):
    logger = logging.getLogger(__name__)
    total_num_instances = sum(res['total_num_instances'])
    total_num_interactions = sum(res['total_num_interactions'])
    num_failed_objects = sum(res['num_failed_objects'])
    total_iou = sum(res['total_iou'])

    logger.info(
        "Total number of instances: {}, Average num of interactions:{}".format(
            total_num_instances, total_num_interactions / total_num_instances
        )
    )
    logger.info(
        "Total number of failed cases: {}, Avg IOU: {}".format(
            num_failed_objects, total_iou / total_num_instances
        )
    )

    # 'res['dataset_iou_list'] is a list of dicts which has to be merged into a single dict
    dataset_iou_list = {}
    for _d in res['dataset_iou_list']:
        dataset_iou_list.update(_d)

    NOC_80, NFO_80, IOU_80 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.80)
    NOC_85, NFO_85, IOU_85 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.85)
    NOC_90, NFO_90, IOU_90 = get_summary(dataset_iou_list, max_clicks=max_interactions, iou_thres=0.90)

    row = [model_name,
           NOC_80, NOC_85, NOC_90, NFO_80, NFO_85, NFO_90, IOU_80, IOU_85, IOU_90,
           total_num_instances,
           max_interactions]

    save_stats_path = os.path.join("./output/evaluation", f'{dataset_name}.txt')
    os.makedirs("./output/evaluation", exist_ok=True)
    if not os.path.exists(save_stats_path):
        header = ["model",
                  "NOC_80", "NOC_85", "NOC_90", "NFO_80","NFO_85","NFO_90","IOU_80","IOU_85", "IOU_90",
                  "#samples","#clicks"]
        with open(save_stats_path, 'w') as f:
            writer = csv.writer(f, delimiter= "\t")
            writer.writerow(header)
    else:
        with open(save_stats_path, 'a') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(row)

    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["dataset", "NOC_80", "NOC_85", "NOC_90", "NFO_80", "NFO_85", "NFO_90", "#samples",
                         "#clicks"]
    table.add_row(
        [dataset_name, NOC_80, NOC_85, NOC_90, NFO_80, NFO_85, NFO_90, total_num_instances, max_interactions])

    print(table)


def log_multi_instance(res, max_interactions, dataset_name, model_name, iou_threshold=0.85, save_stats_summary=True):
    logger = logging.getLogger(__name__)
    total_num_instances = sum(res['total_num_instances'])
    total_num_interactions = sum(res['total_num_interactions'])
    num_failed_objects = sum(res['num_failed_objects'])
    total_iou = sum(res['total_iou'])

    logger.info(
        "Total number of instances: {}, Average num of interactions:{}".format(
            total_num_instances, total_num_interactions / total_num_instances
        )
    )
    logger.info(
        "Total number of failed cases: {}, Avg IOU: {}".format(
            num_failed_objects, total_iou / total_num_instances
        )
    )

    # header = ['Model Name', 'IOU_thres', 'Avg_NOC', 'NOF', "Avg_IOU", "max_num_iters", "num_inst"]
    NOC = np.round(total_num_interactions / total_num_instances, 2)
    NCI = sum(res['avg_num_clicks_per_images']) / len(res['avg_num_clicks_per_images'])
    NFI = len(res['failed_images_ids'])
    Avg_IOU = np.round(total_iou / total_num_instances, 4)
    row = [model_name, NCI, NFI, NOC, num_failed_objects, Avg_IOU, iou_threshold, max_interactions, total_num_instances]

    save_stats_path = os.path.join("./output/evaluation", f'{dataset_name}.txt')
    os.makedirs("./output/evaluation", exist_ok=True)
    with open(save_stats_path, 'a') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)

    if save_stats_summary:
        summary_stats = {}
        summary_stats["dataset"] = dataset_name
        summary_stats["model"] = model_name
        summary_stats["iou_threshold"] = iou_threshold
        summary_stats["failed_images_counts"] = NFI
        summary_stats["avg_over_total_images"] = NCI
        summary_stats["Avg_NOC"] = NOC
        summary_stats["Avg_IOU"] = np.round(total_iou / total_num_instances, 4)
        summary_stats["num_failed_objects"] = num_failed_objects
        summary_stats["failed_images_ids"] = res['failed_images_ids']
        summary_stats["failed_objects_areas"] = res['failed_objects_areas']
        summary_stats["avg_num_clicks_per_images"] = np.mean(res['avg_num_clicks_per_images'])
        summary_stats["total_computer_time"] = res['total_compute_time_str']
        summary_stats["time_per_intreaction_tranformer_decoder"] = np.mean(
            res['time_per_intreaction_tranformer_decoder']
        )
        summary_stats["time_per_image_features"] = np.mean(res['time_per_image_features'])
        summary_stats["time_per_image_annotation"] = np.mean(res['time_per_image_annotation'])

        save_summary_path = os.path.join(f"./output/evaluations/{dataset_name}")
        os.makedirs(save_summary_path, exist_ok=True)
        stats_file = os.path.join(save_summary_path,
                                  f"{model_name}_{max_interactions}_max_click_th_final_updated_time_summary.pickle")

        import pickle
        with open(stats_file, 'wb') as handle:
            pickle.dump(summary_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        return None

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        # evaluation_dataset = cfg.DATASETS.TEST[0]
        if dataset_name in ["GrabCut", "Berkeley", "coco_Mval", "davis_single_inst"]:
            mapper = COCOMvalDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif dataset_name in ["davis_2017_val", "sbd_single_inst","sbd_multi_insts"]:
            mapper = DAVIS17DetmClicksDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        else:
            return None
        
    @classmethod
    def build_train_loader(cls, cfg):
        datset_mapper_name = cfg.INPUT.DATASET_MAPPER_NAME
        from mask2former.utils.equal_num_instances_batch import build_detection_train_loader_equal
        if datset_mapper_name == "multi_instances_clicks_stuffs_mq":
            mapper = COCOMultiInstStuffMultiQueriesClicksDatasetMapper(cfg,True)
            return build_detection_train_loader_equal(cfg, mapper=mapper)
        elif datset_mapper_name == "single_instance_clicks_stuffs_mq":
            mapper = COCOSingleInstMultiQueriesStuffClicksDatasetMapper(cfg,True)
            return build_detection_train_loader_equal(cfg, mapper=mapper)
        elif datset_mapper_name == "coco_lvis_single_inst_stuff_mq":
            mapper = COCOLVISSingleInstMQClicksDatasetMapper(cfg,True)
            return build_detection_train_loader_equal(cfg, mapper=mapper)
        elif datset_mapper_name == "coco_lvis_multi_insts_stuff_mq":
            mapper = COCOLVISMultiInstMQClicksDatasetMapper(cfg,True)
            return build_detection_train_loader_equal(cfg, mapper=mapper)
        elif datset_mapper_name == "coco_lvis_multi_insts_stuff_coords_mq":
            mapper = COCOLVISMultiInstMQCoordsDatasetMapper(cfg,True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            max_interactions = 10
            iou_threshold = 0.85
            data_loader = cls.build_test_loader(cfg, dataset_name)
            
            evaluator =None
            if dataset_name in ["coco_2017_val", "davis_2017_val", "sbd_multi_insts"]:
                model_name = cfg.MODEL.WEIGHTS.split("/")[-2]
                from mask2former.evaluation.multi_instance_evaluation import evaluate
                results_i = evaluate(model, data_loader,cfg, dataset_name)
                results_i = comm.gather(results_i, dst=0)  # [res1:dict, res2:dict,...]
                if comm.is_main_process():
                    # sum the values with same keys
                    assert len(results_i) > 0
                    res_gathered = results_i[0]
                    results_i.pop(0)
                    for _d in results_i:
                        for k in _d.keys():
                            res_gathered[k] += _d[k]
                    log_multi_instance(res_gathered, max_interactions=max_interactions,
                                        dataset_name=dataset_name, model_name=model_name)
            else:
                max_interactions = 20
                iou_threshold = [0.90]
                for iou in iou_threshold:
                    for s in range(3):
                        model_name = cfg.MODEL.WEIGHTS.split("/")[-2] + f"_S{s}"
                        results_i = get_avg_noc(model, data_loader, cfg, iou_threshold = iou,
                                                dataset_name=dataset_name,sampling_strategy=s,
                                                max_interactions=max_interactions)
                        results_i = comm.gather(results_i, dst=0)  # [res1:dict, res2:dict,...]
                        if comm.is_main_process():
                            # sum the values with same keys
                            assert len(results_i)>0
                            res_gathered = results_i[0]
                            results_i.pop(0)
                            for _d in results_i:
                                for k in _d.keys():
                                    res_gathered[k] += _d[k]
                            # res_gathered = dict(
                            #     functools.reduce(operator.iconcat,
                            #                      map(collections.Counter, results_i))
                            # )
                            # print_csv_format(res_gathered)
                            log_single_instance(res_gathered, max_interactions=max_interactions,
                                                dataset_name=dataset_name, model_name=model_name)

        return {}

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # if args.eval_only:
    #     cfg.SEED = 46699430
    # cfg.OUTPUT_DIR = "./all_data/new_models/class_agnostic"
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["NCCL_P2P_DISABLE"] = str(1)
    # import debugpy

    # debugpy.listen(5678)
    # print("Waiting for debugger")
    # debugpy.wait_for_client()
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        # model = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)._load_file(
        #     cfg.MODEL.WEIGHTS
        # )

        # model = model['model']
        # model.eval()
        # model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])
        # model = torch.load(cfg.MODEL.WEIGHTS)['model']
        res = Trainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        return res

    # import pdb; pdb.set_trace()
    
    trainer = Trainer(cfg)
    # breakpoint()
    trainer.resume_or_load(resume=args.resume)
    #Save custom config
    # with open(cfg.OUTPUT_DIR + "/iterative_cfg.yaml", "w") as f: 
    #     f.write(cfg.dump())
    # trainer.resume_or_load(resume=True)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
