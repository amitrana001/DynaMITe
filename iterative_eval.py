# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
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
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
# from mask2former.evaluation.num_clicks_eval import get_avg_noc
from mask2former.evaluation.clicks_evaluation_cc import get_avg_noc

from detectron2.evaluation import (
    DatasetEvaluator,
    print_csv_format,
    verify_results,
)

from mask2former.evaluation.iterative_evaluator import iterative_inference_on_dataset
# MaskFormer
from mask2former import (
    COCOEvalDetmClicksDatasetMapper,
    DAVIS17DetmClicksDatasetMapper,
    COCOInteractiveClicksDatasetMapper,
    COCOEvaluationDatasetMapper,
    COCOMvalDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)


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
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        val_data = cfg.DATASETS.TEST[0]
        if cfg.DATASETS.TEST[0] == 'GrabCut' or cfg.DATASETS.TEST[0] == 'Berkeley' or cfg.DATASETS.TEST[0] == 'coco_Mval':
            from mask2former.data.datasets.register_grabcut import register_grabcut
            from mask2former.data.datasets.register_coco_mval import register_coco_mval
            # register_berkeley()
            # print("BerKely Dataset")
            print(dataset_name)
            mapper = COCOMvalDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif cfg.DATASETS.TEST[0] == "davis_single_inst":
            print("davis dataset mapper")
            from mask2former.data.datasets import register_davis_single_inst
            print(dataset_name)
            mapper = COCOMvalDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif (val_data == "davis_2017_val") or (val_data == "sbd_single_inst") or (val_data == "sbd_multi_insts"):
            print("davis dataset mapper")
            from mask2former.data.datasets import register_davis17
            from mask2former.data.datasets import sbd_single_inst
            print(dataset_name)
            mapper = DAVIS17DetmClicksDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif val_data == "coco_2017_val":
            print("interactive dataset mapper")
            print(dataset_name)
            mapper = COCOEvalDetmClicksDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "interactive_coco_instance_lsj":
            print("interactive dataset mapper")
            print(dataset_name)
            mapper = COCOEvaluationDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_2017_val":
            print("interactive dataset mapper")
            print(dataset_name)
            mapper = COCOEvalDetmClicksDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_Mval":
            from mask2former.data.datasets.register_coco_mval import register_coco_mval
            # register_coco_mval()
            print("COCO MVal Dataset")
            print(dataset_name)
            mapper = COCOMvalDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "Berkeley":
            from mask2former.data.datasets.register_berkeley import register_berkeley
            # register_berkeley()
            print("BerKely Dataset")
            print(dataset_name)
            assert dataset_name == 'Berkeley'
            mapper = COCOMvalDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "GrabCut":
            from mask2former.data.datasets.register_grabcut import register_grabcut
            # register_berkeley()
            print("BerKely Dataset")
            print(dataset_name)
            mapper = COCOMvalDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "interactive_coco_instance_clicks":
            print("interactive clicks dataset mapper")
            print(dataset_name)
            mapper = COCOInteractiveClicksDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        else:
            return None
        
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
        print(cfg.DATASETS.TEST)
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
           
            evaluator =None
            # max_interactions =  cfg.ITERATIVE.TEST.MAX_NUM_INTERACTIONS 
            # iou_threshold = cfg.ITERATIVE.TEST.IOU_THRESHOLD
            do_all = True
            if do_all:
                iou_thresholds = [0.85, 0.90]
                max_num_clicks = [20]
                do_post_process = [False, True]
                for iou_threshold in iou_thresholds:
                    for max_interactions in max_num_clicks:
                                results_i = get_avg_noc(model, data_loader, cfg, evaluator,
                                                iou_threshold, max_interactions, is_post_process=False)
            else:
                results_i = get_avg_noc(model, data_loader, cfg, evaluator,
                                        iou_threshold, max_interactions, is_post_process=False)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

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
    cfg.OUTPUT_DIR = "./all_data/evaluations/grabcut/"
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def main(args):
    cfg = setup(args)

    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["NCCL_P2P_DISABLE"] = str(1)
    
    if args.eval_only:
        # cfg.defrost()
        # cfg.DATASETS.TEST = ("coco_Mval",)
        # cfg.freeze()
        model = Trainer.build_model(cfg)
        # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=args.resume
        # )

        # model = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)._load_file(
        #     cfg.MODEL.WEIGHTS
        # )

        # model = model['model']
        model.eval()
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # import pdb; pdb.set_trace()
    import debugpy

    debugpy.listen(5678)
    print("Waiting for debugger")
    debugpy.wait_for_client()
    trainer = Trainer(cfg)
    breakpoint()
    trainer.resume_or_load(resume=args.resume)
    #Save custom config
    with open("./all_data/evaluations/grabcut/iterative_cfg.yaml", "w") as f: 
        f.write(cfg.dump())
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
