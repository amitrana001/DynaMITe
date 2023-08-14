# DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer

**[Computer Vision Group, RWTH Aachen University](https://www.vision.rwth-aachen.de/)**

[Amit Kumar Rana](https://amitrana001.github.io/), [Sabarinath Mahadevan](https://www.vision.rwth-aachen.de/person/218/), [Alexander Hermans](https://www.vision.rwth-aachen.de/person/10/), [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/)

[[`Paper`](https://arxiv.org/pdf/2304.06668.pdf)] [[`ArXiv`](https://arxiv.org/abs/2304.06668)] [[`Project-Page`](https://sabarim.github.io/dynamite/)] [[`BibTeX`](#citing-segment-anything)]
<div align="center">
  <img src="https://github.com/amitrana001/DynaMITe/blob/final/assets/arch.png" width="100%" height="100%"/>
</div><br/>

## Interactive Segmentation Demo

<div align="center">
  <img src="https://github.com/amitrana001/DynaMITe/blob/final/assets/demo.png" width="100%" height="100%"/>
</div><br/>

1. Pick a model and its config file from
  [model zoo](assets/MODEL_ZOO.md),
  for example, `configs/coco_lvis/swin/dynamite_swin_tiny_bs32_ep50.yaml`.
2. We provide `demo.py` that is able to demo builtin configs. Run it with:
```
python demo.py --config-file configs/coco_lvis/swin/dynamite_swin_tiny_bs32_ep50.yaml \
  --model-weights /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify 'model-weights' to a model from model zoo for evaluation.
This command will open an OpenCV window where you can select any image and perform interactive segementation on it.
All the buttons in the tool are self-explaintory.

## Installation

See [installation instructions](assets/INSTALL.md).

## Datasets
See [Preparing Datasets for DynaMITe](assets/DATASETS.md).

## <a name="GettingStarted"></a>Getting Started

See [Training and Evaluation](assets/GETTING_STARTED.md).

## <a name="Models"></a>Model Checkpoints

Trained models are available for download in the [DynaMITe Model Zoo](assets/MODEL_ZOO.md).

## Citing DynaMITe

```BibTeX
@inproceedings{RanaMahadevan23Arxiv,
      title={DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer},
      author={Rana, Amit and Mahadevan, Sabarinath and Hermans, Alexander and Leibe, Bastian},
      booktitle={ICCV},
      year={2023}
}
```