from mask2former.data.datasets.register_coco_lvis import *
from detectron2.data import DatasetCatalog, MetadataCatalog
d = DatasetCatalog.get("coco_lvis_2017_train")