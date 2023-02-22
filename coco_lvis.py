from mask2former.data.datasets.register_coco_lvis import *
from detectron2.data import DatasetCatalog, MetadataCatalog
import debugpy

# debugpy.listen(5678)
# print("Waiting for debugger")
# debugpy.wait_for_client()
d = DatasetCatalog.get("coco_lvis_2017_train")