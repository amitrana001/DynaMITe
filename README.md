# DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer

**[Computer Vision Group, RWTH Aachen University](https://www.vision.rwth-aachen.de/)**

[Amit Kumar Rana](https://amitrana001.github.io/), [Sabarinath Mahadevan](https://www.vision.rwth-aachen.de/person/218/), [Alexander Hermans](https://www.vision.rwth-aachen.de/person/10/), [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/)

[[`Paper`](https://arxiv.org/pdf/2304.06668.pdf)] [[`ArXiv`](https://arxiv.org/abs/2304.06668)] [[`Project-Page`](https://sabarim.github.io/dynamite/)] [[`BibTeX`](#citing-segment-anything)]
<div align="center">
  <img src="https://github.com/amitrana001/DynaMITe/blob/final/assets/arch.png" width="100%" height="100%"/>
</div><br/>

## Interactive Segmentation Demo

<div align="center">
  <img src="https://github.com/amitrana001/DynaMITe/blob/final/assets/cakes.gif" width="420"/>
</div><br/>

1. Pick a model and its config file from
  model checkpoints,
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
    <li>Clicks management</li>
    <ul>
        <li><i>add instance </i> button to add a new instance; a button for the new instance would be created with the same color as the color of the instance mask. </li> 
        <li><i>bg clicks</i> button to add background clicks.</li>
        <li><i>reset clicks</i> button to remove all clicks and instances.</li>
    </ul>
    <li>Visualisation parameters</li>
    <ul>
        <li><i>show masks only </i> button to visualize only the masks without point clicks. </li> 
        <li><i>Alpha blending coefficient</i> slider adjusts the intensity of all predicted masks.</li>
        <li><i>Visualisation click radius</i> slider adjusts the size of red and green dots depicting clicks.</li>
    </ul>
</ul>
</details>

## <a name="Models"></a>Model Checkpoints

We provide pretrained models with different backbones for interactive segmentation.

You can find the model weights and evaluation results in the tables below. Although we provide hyperlinks against the respective table entries, all models are trained in the multi-instance setting, and are applicable for both single and multi-instance settings.

<table>
    <thead align="center">
        <tr>
            <th align="center", colspan="14">Multi-instance Interactive Segmentation</th>
        </tr>
        <tr>
            <th rowspan="2">Model</th>
            <th rowspan="2">Strategy</th>
            <th colspan="4">COCO</th>
            <th colspan="4">SBD</th>    
            <th colspan="4">DAVIS</th>
        </tr>
        <tr>
            <td>NCI<br>85%</td>
            <td>NFO<br>85%</td>
            <td>NFI<br>85%</td>
            <td>mIoU<br>85%</td>
            <td>NCI<br>85%</td>
            <td>NFO<br>85%</td>
            <td>NFI<br>85%</td>
            <td>mIoU<br>85%</td>
            <td>NCI<br>85%</td>
            <td>NFO<br>85%</td>
            <td>NFI<br>85%</td>
            <td>mIoU<br>85%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td align="left", rowspan="3"><a href="https://drive.google.com/file/d/1DCDiPv9Cr3nlKoUyxzkwK6pPnalVbfjn/view?usp=drive_link">Segformer-B0</a></td>
            <td>best</td>
            <td>6.13</td>
            <td>15219</td>
            <td>2485</td>
            <td>81.3</td>
            <td>2.83</td>
            <td>655</td>
            <td>342</td>
            <td>90.2</td>
            <td>3.29</td>
            <td>546</td>
            <td>364</td>
            <td>87.5</td>
        </tr>
        <tr>
            <td>random</td>
            <td>6.04</td>
            <td>12986</td>
            <td>2431</td>
            <td>84.9</td>
            <td>2.76</td>
            <td>528</td>
            <td>313</td>
            <td>90.6</td>
            <td>3.27</td>
            <td>549</td>
            <td>356</td>
            <td>87.9</td>
        </tr>
         <tr>
            <td>worst</td>
            <td>6.02</td>
            <td>19758</td>
            <td>2414</td>
            <td>83.0</td>
            <td>2.75</td>
            <td>842</td>
            <td>315</td>
            <td>90.3</td>
            <td>3.25</td>
            <td>707</td>
            <td>354</td>
            <td>87.1</td>
        </tr>
        <tr>
            <td align="left", rowspan="3"><a href="https://drive.google.com/file/d/1RllYat-UWD9oQ4HN6pmouT3vtVf9It4x/view?usp=drive_link">Swin-Large</a></td>
            <td>best</td>
            <td>5.80</td>
            <td>13876</td>
            <td>2305</td>
            <td>82.4</td>
            <td>2.47</td>
            <td>497</td>
            <td>266</td>
            <td>90.7</td>
            <td>3.06</td>
            <td>483</td>
            <td>330</td>
            <td>88.4</td>
        </tr>
        <tr>
            <td>random</td>
            <td>5.70</td>
            <td><ins>11958</ins></td>
            <td><ins>2242</ins></td>
            <td><ins>85.3</ins></td>
            <td>2.42</td>
            <td><ins>428</ins></td>
            <td><ins>249</ins></td>
            <td><ins>91.0</ins></td>
            <td>3.03</td>
            <td><ins>479</ins></td>
            <td><ins>320</ins></td>
            <td><ins>88.8</ins></td>
        </tr>
        <tr>
            <td>worst</td>
            <td><ins>5.66</ins></td>
            <td>18133</td>
            <td>2242</td>
            <td>83.7</td>
            <td><ins>2.41</ins></td>
            <td>671</td>
            <td>251</td>
            <td>90.8</td>
            <td><ins>2.99</ins></td>
            <td>620</td>
            <td>314</td>
            <td>88.1</td>
        </tr>
    </tbody>
</table>

<table>
    <thead align="center">
        <tr>
            <th align="center", colspan="13">Single-instance Interactive Segmentation</th>
        </tr>
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
            <td>2.13</td>
            <td>2.51</td>
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
            <td>2.07</td>
            <td>2.43</td>
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
            <td>2.04</td>
            <td>2.40</td>
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
            <td>1.94</td>
            <td>2.27</td>
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
            <td><ins>1.83</ins></td>
            <td><ins>2.12</ins></td>
            <td><ins>2.19</ins></td>
            <td><ins>2.88</ins></td>
        </tr>
    </tbody>
</table>

## Installation

See [Installation Instructions](assets/INSTALL.md).

## Datasets
See [Preparing Datasets for DynaMITe](assets/DATASETS.md).

## <a name="GettingStarted"></a>Getting Started

See [Training and Evaluation](assets/GETTING_STARTED.md).

## Reproducibility
We train all the released checkpoints using a fixed seed, mentioned in the corresponding config files for each backbone. We use 16 GPUs with batch size of 32 and initial global learning rate of 0.0001 for training. Each GPU is an NVIDIA A100 Tensor Core GPU with 40 GB. The evaluation is also done on the same GPUs. <br>
Note: different machines will exhibit distinct hardware and software stacks, potentially resulting in minute variations in the outcomes of floating-point operations.

We train the Swin-Tiny model 3 times with different seeds during training and observe the variance in evaluation metrics as follows:

<table>
    <thead align="center">
        <tr>
            <th align="center", colspan="14">Multi-instance Interactive Segmentation</th>
        </tr>
        <tr>
            <th rowspan="2">Model</th>
            <th rowspan="2">Best <br>Strategy</th>
            <th colspan="4">COCO</th>
            <th colspan="4">SBD</th>    
            <th colspan="4">DAVIS</th>
        </tr>
        <tr>
            <td>NCI<br>85%</td>
            <td>NFO<br>85%</td>
            <td>NFI<br>85%</td>
            <td>mIoU<br>85%</td>
            <td>NCI<br>85%</td>
            <td>NFO<br>85%</td>
            <td>NFI<br>85%</td>
            <td>mIoU<br>85%</td>
            <td>NCI<br>85%</td>
            <td>NFO<br>85%</td>
            <td>NFI<br>85%</td>
            <td>mIoU<br>85%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td align="left", rowspan="2">Swin-Tiny</td>
            <td>mean</td>
            <td>6.13</td>
            <td>15219</td>
            <td>2485</td>
            <td>81.3</td>
            <td>2.83</td>
            <td>655</td>
            <td>342</td>
            <td>90.2</td>
            <td>3.29</td>
            <td>546</td>
            <td>364</td>
            <td>87.5</td>
        </tr>
        <tr>
            <td>std</td>
            <td>6.04</td>
            <td>12986</td>
            <td>2431</td>
            <td>84.9</td>
            <td>2.76</td>
            <td>528</td>
            <td>313</td>
            <td>90.6</td>
            <td>3.27</td>
            <td>549</td>
            <td>356</td>
            <td>87.9</td>
        </tr>
    </tbody>
</table>

<table>
    <thead align="center">
        <tr>
            <th align="center", colspan="13">Single-instance Interactive Segmentation</th>
        </tr>
        <tr>
            <th rowspan="2">Model<br>Swin-Tiny</th>
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
            <td align="left">mean</a></td>
            <td>1.49</td>
            <td>1.59</td>
            <td>1.37</td>
            <td>2.00</td>
            <td>3.72</td>
            <td>6.26</td>
            <td>3.79</td>
            <td>5.08</td>
            <td>1.95</td>
            <td>2.27</td>
            <td>2.22</td>
            <td>3.08</td>
        </tr>
        <tr>
            <td align="left">std</a></td>
            <td>0.05</td>
            <td>0.08</td>
            <td>0.04</td>
            <td>0.11</td>
            <td>0.04</td>
            <td>0.01</td>
            <td>0.10</td>
            <td>0.10</td>
            <td>0.03</td>
            <td>0.02</td>
            <td>0.08</td>
            <td>0.09</td>
        </tr>
    </tbody>
</table>

## Acknowledgement
The main codebase is built on top of [detectron2](https://github.com/facebookresearch/detectron2) framework and is inspired from [Mask2Fromer](https://github.com/facebookresearch/Mask2Former).

The Interactive segementation tool is modified from [RITM]( https://github.com/saic-vul/ritm_interactive_segmentation).

## Citing DynaMITe

If you use our codebase then please cite the papers mentioned below.

```BibTeX
@inproceedings{RanaMahadevan23Arxiv,
      title={DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer},
      author={Rana, Amit and Mahadevan, Sabarinath and Hermans, Alexander and Leibe, Bastian},
      booktitle={ICCV},
      year={2023}
}

@inproceedings{RanaMahadevan23cvprw,
      title={Clicks as Queries: Interactive Transformer for Multi-instance Segmentation},
      author={Rana, Amit and Mahadevan, Sabarinath and Alexander Hermans and Leibe, Bastian},
      booktitle={CVPRW},
      year={2023}
}
```
