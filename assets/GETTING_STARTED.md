## Getting Started with DynaMITe

This document provides a brief intro of the usage of DynaMITe.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Training

We provide a script `train_net.py`, that is made to train all the configs provided in DynaMITe.

To train a model with "train_net.py", first
setup the corresponding datasets following
[DATASETS.md](DATASETS.md),
then run:
```
python train_net.py --num-gpus 16 \
  --config-file configs/coco_lvis/swin/dynamite_swin_tiny_bs32_ep50.yaml
```

The configs are made for 16-GPU training.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
To train on 1 GPU, you need to figure out learning rate and batch size by yourself:
```
python train_net.py \
  --config-file configs/coco_lvis/swin/dynamite_swin_tiny_bs32_ep50.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
```


### Evaluation
To evaluate a model's performance, use
```
python train_net.py \
  --config-file configs/coco_lvis/swin/dynamite_swin_tiny_bs32_ep50.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file \
  --eval-datasets "(davis_2017_val,sbd_multi_insts)"
  --eval-strategy random
  --seed-id 1
  --max-interactions 10
  --iou-threshold 0.85
  --vis-path /path/to/save_visulization
```
For more options, see `python train_net.py -h`.

<details>
<summary><b>Evaluation argument options</b></summary>
<ul>
    <!-- <li>Visualisation parameters</li> -->
    <!-- <ul> -->
    <li><i>--eval-datasets: </i> Expect a list of names of registered datasets that you want your model to evaluate on. See [Preparing Datasets for DynaMITe](assets/DATASETS.md) for more details.Available options are:</li>
    Single-instnace datasets:
    <ul>
        <li><i>GrabCut</i></li>
        <li><i>Brekeley</i></li>
        <li><i>COCO_Mval</i></li>
        <li><i>davis_single_inst</i></li>
        <li><i>pascal_voc_single</i></li>
        <li><i>sbd_single_inst</i></li>
    </ul> 
    Multi-instance datsets:
    <ul>
        <li><i>coco_2017_val</i></li>
        <li><i>davis_2017_val</i></li>
        <li><i>sbd_multi_insts</i></li>
    </ul>
    <li><i>--eval-strategy: </i> Click sampling strategy for evaluating multi-instance interactive segmentation. Available options are:</li>
    <ul>
        <li><i>best</i></li>
        <li><i>random</i></li>
        <li><i>worst</i></li>
        <li><i>max_dt</i></li>
        <li><i>round_robin</i></li>
        <li><i>wlb</i></li>
    </ul> 
    where wlb: worst with limit, max_dat: maximum distance transform based strategy.
    <li><i>--seed-id: </i> Fixing seed for random evaluation strategy for reproducibility.</li>
    <li><i>--max-interactions: </i> Maximum number of clicks per object/instance. Normally set to 20 for single-instnace and 10 for multi-instance evaluation.</li>
    <li><i>--iou-threshold: </i> Desired IoU threshold for evaluation. Normally set to 0.90 for single-instnace and 0.85 for multi-instance evaluation.</li>
    <li><i>--vis-path: </i> Path to save segmentation visualization after each click. By default, set to None.</li>
</ul>
</details>
