# Tracking Project's Progress

## Tasks TO DO
* New Evaluation Strategy:
    - Start with one click per object
    - Sample one click in one interaction on the largest error region
    - Record the number of switch between objects
    - To get the Efect of clicks on some othe masks/regions
        - Recoed the ious per objects and sampled click's object id info in each iteration
* Getting better results with ResNet50 backbone
* Getting to SoTA for single instance evaluation
    - Record all statistics during evaluation
    - Zoom-in and without zoom-in
* Spatio-temporal positional encodings for multi query setup
* Try multiple positional encodings for queries
    - Learnable static bg queries and positional encodings (= #static_bg_queries)
    - Try fixed sinusodial for queries based on the 2D click coordinates
        - Add to the learnable query for better representation
    - Try out fixed and learnable relative positional encodings
* Loss formulation modification:
    - Based on [['INTERACTIVE IMAGE SEGMENTATION WITH TRANSFORMERS'](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9897542)]

## Design Choices for Query Initializers or Transformer Decoder
- [ x ] Global average pooling across all features scales.
- [ x ] Non-linear projection of query descriptors.
- [ ] Replace transformer decoder with HODOR encoder-decoder network.
- [ ] ViT based encoder (take patches around scribbles, pass them through ViT and get embedding for all patches, accumulate or global pool all features corresponding to one instance).

## Tasks Accomplished
*
   
## List of Relevant Papers
* Mask2Former: [[`paper`](https://arxiv.org/abs/2112.01527)] [[`github`](https://github.com/facebookresearch/Mask2Former)]
* MaskFormer: [[`paper`](https://arxiv.org/abs/2107.06278)] [[`github`](https://github.com/facebookresearch/MaskFormer)]
* DETR: [[`paper`](https://arxiv.org/abs/2005.12872)] [[`github`](https://github.com/facebookresearch/detr)]
* Swin Transformer: [[`paper`](https://arxiv.org/abs/2103.14030)] [[`github`](https://github.com/microsoft/Swin-Transformer)]
* HODOR: [[`paper`](https://arxiv.org/abs/2112.09131)] [[`github`](https://github.com/Ali2500/HODOR)]
* Dynamic Focus-aware Positional Queries: [[`paper`](https://arxiv.org/abs/2204.01244)] [[`github`](https://github.com/ziplab/FASeg)]
* MiVOS: [[`paper`](https://arxiv.org/abs/2103.07941)] [[`github`](https://github.com/hkchengrex/MiVOS)]
* Mask Scoring R-CNN: [[`paper`](https://arxiv.org/pdf/1903.00241)] [[`github`](https://github.com/zjhuang22/maskscoring_rcnn)]


### Features
* A single architecture for panoptic, instance and semantic segmentation.
* Support major segmentation datasets: ADE20K, Cityscapes, COCO, Mapillary Vistas.

## Updates
* Add Google Colab demo.
* Video instance segmentation is now supported! Please check our [tech report](https://arxiv.org/abs/2112.10764) for more details.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for Mask2Former](datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

Replicate web demo and docker image is available here: [![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

## Advanced usage

See [Advanced Usage of Mask2Former](ADVANCED_USAGE.md).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Mask2Former Model Zoo](MODEL_ZOO.md).

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="CitingMask2Former"></a>Citing Mask2Former

If you use Mask2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

## Acknowledgement

Code is largely based on MaskFormer (https://github.com/facebookresearch/MaskFormer).