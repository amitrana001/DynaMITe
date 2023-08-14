# Prepare Datasets for DynaMITe

We register all the interactive segmentation datasets (for single and multi-instnace interactive segmentation) in Detectron2 format where
a dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc). 

This document explains how to setup the interactive segmentation datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`, and how to add new datasets to them.

Note: We train and evalaute all the models class-agnostically so most of the datasets registered for our purpose have meta-data set to None.

The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  lvis/
  GrabCut/
  Berkeley/
  COCO_MVal/
  sbd/
  DAVIS/
  pascal_voc/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

The datasets are registered in detectron2 format using the following files present in dynamite/data/datasets/ .

```
dynamite/data/datasets
    ├── register_berkeley.py
    ├── register_coco_lvis.py
    ├── register_coco_mval.py
    ├── register_davis17.py
    ├── register_davis_single_inst.py
    ├── register_grabcut.py
    ├── register_pascal_voc_single_inst.py
    ├── register_sbd_multi_insts.py
    └── register_sbd_single_inst.py
```
We downloaded all the datasets following the links provided in [RITM github page](https://github.com/SamsungLabs/ritm_interactive_segmentation/tree/master).

## Expected dataset structure for [COCO+LVIS](https://cocodataset.org/#download):

Either download the compiled COCO+LVIS dataset ([coco_lvis_combined.pickle](link)) in detectron2 format or download the [original LVIS images](https://www.lvisdataset.org/dataset) and combined annotations by [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation/tree/master) ([hannotation.pickle](https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/cocolvis_annotation.tar.gz)).
```
DETECTRON2_DATASETS
└── lvis
    ├── train
    │   ├── images
    │   ├── masks
    |   └── hannotation.pickle
    └── coco_lvis_combined.pickle
```

## Expected dataset structure for Evaluation datasets:

The snippet provides the dataset directory structure along with the names of registered datasets.
```
DETECTRON2_DATASETS                    Dataset-Name
├── Berkeley                            "Brekeley"
│   ├── images
│   └── masks
├── GrabCut                             "GrabCut"
│   ├── boundary_GT
│   └── data_GT
├── COCO_MVal                           "COCO_Mval"
│   ├── gt
│   └── img
├── davis                               "davis_single_inst"
│   ├── gt
│   └── img
├── DAVIS                               "davis_2017_val"
|   └── DAVIS-2017-trainval
|       ├── Annotations
|       │   └── 480p
|       ├── ImageSets
|       │   └── 2017
|       |       └── val.txt      
|       └── JPEGImages
|           └── 480p
├── coco                                "coco_2017_val" 
│   ├── annotations
│   │   └── instnaces_val2017.json
│   └── val2017
├── pascal_voc                          "pascal_voc_single" 
│   ├── ImageSets
│   │   └── Segmentation
│   │       └── val.txt
│   ├── JPEGImages
│   └── SegmentationObject
└── sbd                                 "sbd_single_inst"
    └── dataset                         "sbd_multi_insts"        
        ├── img
        ├── inst
        └── val.txt


```

* The single instnace interactive segmentation datasets are regisered as "GrabCut", "Berkeley", "davis_single_inst", "coco_Mval", "sbd_single_inst", "pascal_voc_single".
* The multi-instnace interactive segmentation datasets are regisered as "davis_2017_val","sbd_multi_insts","coco_2017_val".
