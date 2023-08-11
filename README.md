# DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer

**[Computer Vision Group, RWTH Aachen University](https://www.vision.rwth-aachen.de/)**

[Amit Kumar Rana](https://amitrana001.github.io/), [Sabarinath Mahadevan](https://www.vision.rwth-aachen.de/person/218/), [Alexander Hermans](https://www.vision.rwth-aachen.de/person/10/), [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/)

[[`Paper`](https://arxiv.org/pdf/2304.06668.pdf)] [[`ArXiv`](https://arxiv.org/abs/2304.06668)] [[`Project-Page`](https://sabarim.github.io/dynamite/)] [[`BibTeX`](#citing-segment-anything)]
<div align="center">
  <img src="images/arch.png?raw=true" width="100%" height="100%"/>
</div><br/>

DynaMITe consists of a backbone, a feature decoder, and an interactive Transformer. Point features at click
locations at time t are translated into queries which, along with the multi-scale features, are processed by a Transformer
encoder-decoder structure to generate a set of output masks &#8499;<sup>t</sup> for all the relevant objects. Based on Mt, the user provides
a new input click which is in turn used by the interactive Transformer to generate a new set of updated masks &#8499;<sup>t+1</sup>. This
process is then iterated τ times until the masks are fully refined.

## Installation

## <a name="GettingStarted"></a>Getting Started

## Datasets

## <a name="Models"></a>Model Checkpoints

## Citing DynaMITe

```BibTeX
@inproceedings{RanaMahadevan23Arxiv,
      title={DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer},
      author={Rana, Amit and Mahadevan, Sabarinath and Hermans, Alexander and Leibe, Bastian},
      booktitle={ICCV},
      year={2023}
}
```