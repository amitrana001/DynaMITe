# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_hrnet_config

# dataset loading
from .data.dataset_mappers.clicks.coco_lvis_multi_insts_mq_coords_mapper import COCOLVISMultiInstMQCoordsDatasetMapper
from .data.dataset_mappers.eval.evaluation_dataset_mapper import EvaluationDatasetMapper

from .test_time_augmentation import SemanticSegmentorWithTTA
from .dynamite_model import DynamiteModel
