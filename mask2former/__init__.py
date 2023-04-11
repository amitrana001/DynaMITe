# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_hrnet_config

# dataset loading

from .data.dataset_mappers.coco_evaluation_dataset_mapper import COCOEvaluationDatasetMapper
# from .data.dataset_mappers.coco_single_inst_mapper import COCOSingleInstDatasetMapper
from .data.dataset_mappers.coco_mval_dataset_mapper import COCOMvalDatasetMapper
#for instance interactive mapper 
from .data.dataset_mappers.clicks.coco_lvis_multi_insts_stuff_mq_coords_mapper import COCOLVISMultiInstStuffMQCoordsDatasetMapper
from .data.dataset_mappers.clicks.coco_lvis_multi_insts_mq_coords_mapper import COCOLVISMultiInstMQCoordsDatasetMapper
from .data.dataset_mappers.clicks.coco_lvis_single_inst_mq_coords_mapper import COCOLVISSingleInstMQCoordsDatasetMapper
from .data.dataset_mappers.clicks.coco_lvis_multi_insts_mq_clicks_mapper import COCOLVISMultiInstMQClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_lvis_single_inst_mq_clicks_mapper import COCOLVISSingleInstMQClicksDatasetMapper
# from .data.dataset_mappers.clicks.coco_lvis_2017_clicks_mapper import COCOLVIS2017ClicksDatasetMapper
from .data.dataset_mappers.eval.evaluation_dataset_mapper import EvaluationDatasetMapper
from .data.dataset_mappers.eval.davis_sbd_mq_coords_eval_mapper_V1 import DAVISSBDMQCoordsV1EvalMapper
from .data.dataset_mappers.coco_mval_dataset_mapper_coords_V1 import COCOMvalCoordsV1DatasetMapper
from .data.dataset_mappers.eval.coco_Mval_eval_multi_insts_datamapper import COCOMvalMultiInstsDatasetMapper
from .data.dataset_mappers.eval.davis17_scribbles_datamapper import DAVIS17ScribblesDatasetMapper
from .data.dataset_mappers.eval.davis17_deterministic_clicks_mapper import DAVIS17DetmClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_multi_inst_multi_queries_stuff_clicks_mapper import COCOMultiInstStuffMultiQueriesClicksDatasetMapper
from .data.dataset_mappers.clicks.coco_single_inst_multi_queries_stuff_clicks_mapper import COCOSingleInstMultiQueriesStuffClicksDatasetMapper
from .data.dataset_mappers.eval.coco_eval_mq_coords_mapper import COCOEvalMQCoordsMapper
from .data.dataset_mappers.coco_mval_dataset_mapper_coords import COCOMvalCoordsDatasetMapper
from .data.dataset_mappers.eval.davis_sbd_mq_coords_eval_mapper import DAVISSBDMQCoordsEvalMapper

from .test_time_augmentation import SemanticSegmentorWithTTA
from .iterative_m2f_model import IterativeMask2Former
from .iterative_m2f_mq_model import IterativeMask2FormerMQ
from .spatio_temp_m2f_mq_model import SpatioTempMask2FormerMQ
