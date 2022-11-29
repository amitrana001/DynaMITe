# Tracking Project's Progress

## Tasks TO DO
* Modify code to also include the case with no bg_mask (more practical from interaction's perspective).
* Currently Interactive Datamapper filters out the image with no gt_masks:
    * Implement Custom DataLoader to filter out that.
* Current Scribbles are too fine (lot of scribbles plus very accurate near boundaries):
    * Think of taking only 1 scribble per fg_mask
* Include dynamic focus-aware positional queries in transformer decoder.
* Regress the loss for background mask also:
    * Take max of logits of background masks from all scribbles
    * OR union of all background masks
    * Include bg_mask in datamapper OR calculate it in the decoder itself (during loss) 
* Modify demo GUI to include multiple instances

## Design Choices for Query Initializers or Transformer Decoder
- [ x ] Global average pooling across all features scales.
- [ x ] Non-linear projection of query descriptors.
- [ ] Replace transformer decoder with HODOR encoder-decoder network.
- [ ] ViT based encoder (take patches around scribbles, pass them through ViT and get embedding for all patches, accumulate or global pool all features corresponding to one instance).

## Tasks Accomplished
* End-to-end pipeline for interactive instances segmentation:
    * Interactive Datamapper which generates:
        * Foreground scribbles for each gt_mask.
        * Background scribbles from the rest of the image.
        * Modified MiVOS's Scribble2Mask code to generate scribbles.
    * Query initializer based on HODOR's query initializer.
    * Not Hungratian Matcher required:
        * Matching between predicted masks and gt_mask known based on queries input.
* Trained original Mask2former for coco instance segmentation:
    * Used 1 GPU with batch size 8 and learning rate scaled linearly to the original.
    * Values of APs and ARs were off by a margin of approx. ~2. 
* Implemented pipeline for iterative and interactive inference
* (TO DO -> Accomplished) Currently only initializing queries based on the largest image features:
    * Also think of initializing queries using all feature levels
* (TO DO -> Accomplished) Write a demo script which takes images and scribbles as input and gives the output.
* (TO DO -> Accomplished)Implement a GUI for demonstration:
    * Which takes scribbleds from user's interaction.

    
## List of Relevant Papers
* Mask2Former: [[`paper`](https://arxiv.org/abs/2112.01527)] [[`github`](https://github.com/facebookresearch/Mask2Former)]
* MaskFormer: [[`paper`](https://arxiv.org/abs/2107.06278)] [[`github`](https://github.com/facebookresearch/MaskFormer)]
* DETR: [[`paper`](https://arxiv.org/abs/2005.12872)] [[`github`](https://github.com/facebookresearch/detr)]
* Swin Transformer: [[`paper`](https://arxiv.org/abs/2103.14030)] [[`github`](https://github.com/microsoft/Swin-Transformer)]
* HODOR: [[`paper`](https://arxiv.org/abs/2112.09131)] [[`github`](https://github.com/Ali2500/HODOR)]
* Dynamic Focus-aware Positional Queries: [[`paper`](https://arxiv.org/abs/2204.01244)] [[`github`](https://github.com/ziplab/FASeg)]
* MiVOS: [[`paper`](https://arxiv.org/abs/2103.07941)] [[`github`](https://github.com/hkchengrex/MiVOS)]
* Mask Scoring R-CNN: [[`paper`](https://arxiv.org/pdf/1903.00241)] [[`github`](https://github.com/zjhuang22/maskscoring_rcnn)]
