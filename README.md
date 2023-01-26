# DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer

<div align="center">
  <img src="https://github.com/amitrana001/DynaMITe/blob/main/images/teasure-image.png" width="100%" height="100%"/>
</div><br/>

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


