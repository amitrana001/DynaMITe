## Getting Started with DynaMITe

This document provides a brief intro of the usage of DynaMITe.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Interactive Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](MODEL_ZOO.md),
  for example, `configs/coco_lvis/swin/dynamite_swin_tiny_bs32_ep50.yaml`.
2. We provide `demo.py` that is able to demo builtin configs. Run it with:
```

python demo.py --config-file configs/coco_lvis/swin/dynamite_swin_tiny_bs32_ep50.yaml \
  --model-weights /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify 'model-weights' to a model from model zoo for evaluation.
This command will open an OpenCV window where you can select any image and perform interactive segementation on it.
All the buttons in the tool are self-explaintory.


### Training

We provide a script `train_net.py`, that is made to train all the configs provided in DynaMITe.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md),
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
  --vis-path /path/to/save_visulization
```
For more options, see `python train_net.py -h`.