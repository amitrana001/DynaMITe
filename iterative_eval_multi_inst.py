# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import csv
from detectron2.config import CfgNode as CN
import numpy as np
import argparse
import datetime
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
from dynamite.evaluation.single_instance_evaluation import get_avg_noc

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import (
    DatasetEvaluator,
)

# MaskFormer
from dynamite import (
    EvaluationDatasetMapper,
)

from dynamite import (
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    add_hrnet_config
)

from dynamite.evaluation.eval_utils import log_single_instance, log_multi_instance

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

        assert dataset_name in ["davis_2017_val", "sbd_multi_insts","coco_2017_val"]

        mapper = EvaluationDatasetMapper(cfg,False,dataset_name)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        
    @classmethod
    def build_train_loader(cls, cfg):
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
        max_interactions =10
        # dataset_name = "davis_2017_val"
        seed_id = cfg.EVALUATION_STRATEGY.SEED_ID
        normalize_time = cfg.EVALUATION_STRATEGY.NORMALIZE_TIME 
        eval_strategy = cfg.EVALUATION_STRATEGY.TYPE
        if eval_strategy in ["random", "best", "worst"]:
            from dynamite.inference.random_best_worst import evaluate
        elif eval_strategy == "max_dt":
            from dynamite.inference.max_dt import evaluate
        elif eval_strategy == "wlb":
            from dynamite.inference.wlb import evaluate
        elif eval_strategy == "round_robin":
            from dynamite.inference.round_robin import evaluate

        model_name = cfg.MODEL.WEIGHTS.split("/")[-2]
        model_name += f"_{eval_strategy}_V1"
        if eval_strategy == "random":
            model_name+= f"_seed_{seed_id}"
    
        for idx,dataset_name in enumerate(["davis_2017_val","sbd_multi_insts","coco_2017_val"]):
            # if cfg.ITERATIVE.TRAIN.USE_ARGMAX:
            #     model_name+="_argmax"
            # eval_strategy = cfg.EVALUATION_STRATEGY.TYPE
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # if eval_strategy == "random":
            #     model_name+= f"_seed_{seed_id}"
            
            results_i = evaluate(model, data_loader,cfg, dataset_name, sampling_strategy=1,
                                iou_threshold = 0.85, max_interactions = 10,
                                eval_strategy = eval_strategy,seed_id=seed_id,normalize_time=True)
            results_i = comm.gather(results_i, dst=0)  # [res1:dict, res2:dict,...]
            if comm.is_main_process():
                # sum the values with same keys
                assert len(results_i) > 0
                res_gathered = results_i[0]
                results_i.pop(0)
                for _d in results_i:
                    for k in _d.keys():
                        res_gathered[k] += _d[k]
                log_multi_instance(res_gathered, max_interactions=10,
                                dataset_name=dataset_name, model_name=model_name,
                                sampling_strategy=1)
        
    
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
    cfg.EVALUATION_STRATEGY = CN()
    cfg.EVALUATION_STRATEGY.TYPE = args.eval_strategy
    cfg.EVALUATION_STRATEGY.SEED_ID = args.seed_id
    cfg.EVALUATION_STRATEGY.NORMALIZE_TIME = args.normalize_time

    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_hrnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
   
    eval_dataset = args.eval_dataset
    cfg.DATASETS.TEST = (eval_dataset,)
    cfg.ITERATIVE.TRAIN.USE_ARGMAX = True
    
    # cfg.EVALUATION_STRATEGY = CN()
    cfg.EVALUATION_STRATEGY.TYPE = args.eval_strategy
    cfg.EVALUATION_STRATEGY.SEED_ID = args.seed_id
    cfg.EVALUATION_STRATEGY.NORMALIZE_TIME = args.normalize_time
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    # import debugpy
    # debugpy.listen(5678)
    # print("Waiting for debugger")
    # debugpy.wait_for_client()

    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def default_argument_parser(epilog=None):
    import sys
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:

    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--normalize-time", action="store_true", help="normalize time coordinate")
    parser.add_argument("--eval-dataset", type=str, default="davis_2017_val", help="perform evaluation on given datset")
    parser.add_argument("--eval-strategy", type=str, default="worst", help="evaluation strategy")
    parser.add_argument("--seed-id", type=int, default=0, help="seed id for random evaluation")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

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
