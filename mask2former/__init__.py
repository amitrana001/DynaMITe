# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_interactive_distractor_object_dataset_mapper import COCOInteractiveDistractorDatasetMapper
from .data.dataset_mappers.coco_interactive_clicks_dataset_mapper import COCOInteractiveClicksDatasetMapper
#for interactive datamapper
from .data.dataset_mappers.coco_panoptic_interactive_dataset_mapper import COCOPanopticInteractiveDatasetMapper
from .data.dataset_mappers.coco_evaluation_dataset_mapper import COCOEvaluationDatasetMapper
from .data.dataset_mappers.coco_single_inst_mapper import COCOSingleInstDatasetMapper
from .data.dataset_mappers.coco_mval_dataset_mapper import COCOMvalDatasetMapper
#for instance interactive mapper 
from .data.dataset_mappers.clicks.coco_lvis_multi_insts_mq_clicks_mapper import COCOLVISMultiInstMQClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_lvis_single_inst_mq_clicks_mapper import COCOLVISSingleInstMQClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_lvis_2017_clicks_mapper import COCOLVIS2017ClicksDatasetMapper
from .data.dataset_mappers.eval.coco_Mval_eval_multi_insts_datamapper import COCOMvalMultiInstsDatasetMapper
from .data.dataset_mappers.eval.davis17_scribbles_datamapper import DAVIS17ScribblesDatasetMapper
from .data.dataset_mappers.eval.davis17_deterministic_clicks_mapper import DAVIS17DetmClicksDatasetMapper
from .data.dataset_mappers.clicks.lvis_multi_inst_clicks_dataset_mapper import LVISMultiInstClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_single_inst_stuff_clicks_mapper import COCOSingleInstStuffClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_single_inst_clicks_mapper import COCOSingleInstClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_multi_inst_clicks_dataset_mapper import COCOMultiInstClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_all_inst_clicks_dataset_mapper import COCOAllInstClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_multi_inst_stuff_clicks_mapper import COCOMultiInstStuffClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_multi_inst_multi_queries_stuff_clicks_mapper import COCOMultiInstStuffMultiQueriesClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_single_inst_multi_queries_stuff_clicks_mapper import COCOSingleInstMultiQueriesStuffClicksDatasetMapper
from .data.dataset_mappers.eval.coco_eval_deterministic_clicks_dataset_mapper import COCOEvalDetmClicksDatasetMapper
from .data.dataset_mappers.coco_instance_interactive_dataset_mapper import COCOInstanceInteractiveDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)


# models
# from .maskformer_model import MaskFormer
# from .iterative_maskformer_model import IterativeMaskFormer
# from .new_iterative_maskformer_model import NewIterativeMaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA
from .iterative_m2f_model import IterativeMask2Former
from .iterative_m2f_mq_model import IterativeMask2FormerMQ

# evaluation
# from .evaluation.instance_evaluation import InstanceSegEvaluator
# from .evaluation.num_clicks_eval import get_avg_noc
