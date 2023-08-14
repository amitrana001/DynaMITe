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

<details>
<summary><b>Interactive segmentation options</b></summary>
<ul>
    <li>Visualisation parameters</li>
    <ul>
        <li><i>Prediction threshold</i> slider adjusts the threshold for binarization of probability map for the current object.</li> 
        <li><i>Alpha blending coefficient</i> slider adjusts the intensity of all predicted masks.</li>
        <li><i>Visualisation click radius</i> slider adjusts the size of red and green dots depicting clicks.</li>
    </ul>
</ul>
</details>

## Installation

See [installation instructions](assets/INSTALL.md).

## Datasets
See [Preparing Datasets for DynaMITe](assets/DATASETS.md).

## <a name="GettingStarted"></a>Getting Started

See [Training and Evaluation](assets/GETTING_STARTED.md).

## <a name="Models"></a>Model Checkpoints

We provide pretrained models with different backbones for interactive segmentation.

You can find model weights and evaluation results in the tables below:

<table>
    <thead align="center">
        <tr>
            <th rowspan="2">Model</th>
            <th colspan="2">GrabCut</th>
            <th colspan="2">Berkeley</th>
            <th colspan="2">SBD</th>    
            <th colspan="2">DAVIS</th>
            <th colspan="2">Pascal<br>VOC</th>
            <th colspan="2">COCO<br>MVal</th>
        </tr>
        <tr>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td align="left"><a href="https://drive.google.com/file/d/1uJkKR7FB6I291-94ToW6hjXtiOmdOfeP/view?usp=drive_link">Resent50</a></td>
            <td>1.62</td>
            <td>1.82</td>
            <td>1.47</td>
            <td>2.19</td>
            <td>3.93</td>
            <td>6.56</td>
            <td>4.10</td>
            <td>5.45</td>
            <td>--</td>
            <td>--</td>
            <td>2.36</td>
            <td>3.20</td>
        </tr>
        <tr>
            <td align="left"><a href="https://drive.google.com/file/d/1xMM8Xn0cn5prajAnJB0EpJcyuESfok_h/view?usp=drive_link">HRNet32</a></td>
            <td>1.62</td>
            <td>1.68</td>
            <td>1.46</td>
            <td>2.04</td>
            <td>3.83</td>
            <td>6.35</td>
            <td>3.83</td>
            <td>5.20</td>
            <td>--</td>
            <td>--</td>
            <td>2.35</td>
            <td>3.14</td>
        </tr>
        <tr>
            <td align="left"><a href="https://drive.google.com/file/d/1DCDiPv9Cr3nlKoUyxzkwK6pPnalVbfjn/view?usp=drive_link">Segformer-B0</a></td>
            <td><ins>1.58</ins></td>
            <td><ins>1.68</ins></td>
            <td>1.61</td>
            <td>2.06</td>
            <td>3.89</td>
            <td>6.48</td>
            <td>3.85</td>
            <td>5.08</td>
            <td>--</td>
            <td>--</td>
            <td>2.47</td>
            <td>3.28</td>
        </tr>
        <tr>
            <td align="left"><a href="https://drive.google.com/file/d/14zxbez6JINGQmSo6Vra4S94rwug7JpWK/view?usp=drive_link">Swin-Tiny</a></td>
            <td>1.64</td>
            <td>1.78</td>
            <td>1.39</td>
            <td>1.96</td>
            <td>3.75</td>
            <td>6.32</td>
            <td>3.87</td>
            <td>5.23</td>
            <td>--</td>
            <td>--</td>
            <td>2.24</td>
            <td>3.14</td>
        </tr>
        <tr>
            <td align="left"><a href="https://drive.google.com/file/d/1RllYat-UWD9oQ4HN6pmouT3vtVf9It4x/view?usp=drive_link">Swin-Large</a></td>
            <td>1.62</td>
            <td>1.72</td>
            <td><ins>1.39</ins></td>
            <td><ins>1.90</ins></td>
            <td><ins>3.32</ins></td>
            <td><ins>5.64</ins></td>
            <td><ins>3.80</ins></td>
            <td><ins>5.09</ins></td>
            <td>--</td>
            <td>--</td>
            <td><ins>2.19</ins></td>
            <td><ins>2.88</ins></td>
        </tr>
    </tbody>
</table>

## Citing DynaMITe

```BibTeX
@inproceedings{RanaMahadevan23Arxiv,
      title={DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer},
      author={Rana, Amit and Mahadevan, Sabarinath and Hermans, Alexander and Leibe, Bastian},
      booktitle={ICCV},
      year={2023}
}
```