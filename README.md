
# ColorMAE: Exploring data-independent masking strategies in Masked AutoEncoders

**[Image and Video Understanding Lab, AI Initiative, KAUST](https://ivul.kaust.edu.sa/)**

<blockquote>
<p align="center">
  <a href="https://carloshinojosa.me/">Carlos Hinojosa</a>, 
  <a href="https://sming256.github.io/">Shuming Liu</a>, 
  <a href="https://www.bernardghanem.com/">Bernard Ghanem</a>
  </p>
</blockquote>

![ColorMAE](assets/proposed.png)

<blockquote>
<p align="center">
  <a href="https://arxiv.org/pdf/2407.13036"><code>Paper</code></a> · 
  <a href=""><code>Supplementary Material</code></a> · 
  <a href="https://carloshinojosa.me/project/colormae/"><code>Project</code></a> ·
  <a href="#how-to-cite"><code>BibTeX</code></a>
  </p>
</blockquote>


>Can we enhance MAE performance beyond random masking without relying on input data or incurring additional computational costs? Yes!

We introduce ColorMAE, a simple yet effective **data-independent** method which generates different binary mask patterns by filtering random noise. Drawing inspiration from color noise in image processing, we explore four types of filters to yield mask patterns with different spatial and semantic priors. ColorMAE requires no additional learnable parameters or computational overhead in the network, yet it significantly enhances the learned representations.

## Installation
To get started with ColorMAE, follow these steps to set up the required environment and dependencies. This guide will walk you through creating a Conda environment, installing necessary packages, and setting up the project for use.

1. Clone our repo to your local machine
```shell
git clone https://github.com/carlosh93/ColorMAE.git
cd ColorMAE
```
2. Create conda environment with python 3.10.12 
```shell
conda create --prefix ./venv python=3.10.12
conda activate ./venv
```
3. Install Pytorch 2.0.1 and mmpretrain 1.0.2:
```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U openmim && mim install mmpretrain==1.0.2
```

Note: You can install mmpretrain as a Python package (see above) or from source. Please see [here](https://mmpretrain.readthedocs.io/en/latest/get_started.html#installation).

<!-- 4. `mim install mmengine==0.8.4` -->
<!-- 5. `pip install yapf==0.40.1` -->

## Getting Started

### Setup Environment

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Note: Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the ImageNet-2012 dataset according to the [instruction](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#imagenet). We provide a script and step by step guide [here](data/README.md).

### Download Color Noise Patterns
The following table provides the color noise patterns used in the paper

| Color Noise | Description | Link | Md5 |
|---------------------|-------------|---------------|---------------| 
| Green Noise         | Mid-frequency component of noise. | [Download](https://example.com/white_noise.npy) | `xxx` |
| Blue Noise          | High-frequency component of noise. | [Download](https://example.com/pink_noise.npy) | `xxxx` |
| Purple Noise          | Noise with only high and low-frequency content. | [Download](https://example.com/blue_noise.npy) | `xxx` |
| Red Noise         | Low-frequency component of noise. | [Download](https://example.com/brown_noise.npy) | `xxx` |

You can download these pre-generated color noise patterns and place them in the corresponding folder inside `noise_colors` directory of the project.


## Models and results

In the following table we provide the pretrained and finetuned models with their corresponding results presented in the paper.

### Pretrained models

| Model                                           | Params (M) | Flops (G) |                           Config                           |                                   Download                                   |
| :---------------------------------------------- | :--------: | :-------: | :--------------------------------------------------------: | :--------------------------------------------------------------------------: |
| `colormae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py`   |   111.91   |   16.87   |  [config](configs/colormae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py)  | [model](https://) \| [log](https://) |
| `colormae_vit-base-p16_8xb512-amp-coslr-800e_in1k.py`   |   111.91   |   16.87   |  [config](configs/colormae_vit-base-p16_8xb512-amp-coslr-800e_in1k.py)  | [model](https://) \| [log](https://) |


### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `vit-base-p16_colormae-300e-pre_8xb128-coslr-100e_in1k` | [ColorMAE-G 300-Epochs](https://) |   xx.xx    |   xx.xx   |   82.98  | [config](benchmarks/image_classification/configs/vit-base-p16_8xb128-coslr-100e_in1k.py) |                     N/A                      |
| `vit-base-p16_colormae-400e-pre_8xb128-coslr-100e_in1k` | [ColorMAE-G 800-Epochs](https://) |   xx.xx    |   xx.xx   |   83.57   | [config](benchmarks/image_classification/configs/vit-base-p16_8xb128-coslr-100e_in1k.py) |                      N/A                      |


### Semantic Segmentation on ADE20K

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | mIoU (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `name` | [ColorMAE-G 300-Epochs](https://) |   xx.xx    |   xx.xx   |   45.80   | [config](benchmarks/segmentation/configs/xx.py) |                      N/A                      |
| `name` | [ColorMAE-G 800-Epochs](https://) |   xx.xx    |   xx.xx   |   49.18   | [config](benchmarks/segmentation/configs/xx2.py) |                      N/A                      |

### Object Detection on COCO

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | $AP^{bbox}$ (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `name` | [ColorMAE-G 300-Epochs](https:) |   xx.xx    |   xx.xx   |   48.70   | [config](benchmarks/object_detection/configs/xx.py) |                      N/A                      |
| `name` | [ColorMAE-G 800-Epochs](https://) |   xx.xx    |   xx.xx   |   49.50   | [config](benchmarks/object_detection/configs/xx2.py) |                      N/A                      |

### Using the Models

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('vit-base-p16_mae-300e-pre_8xb128-coslr-100e_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the pretrained model**

```python
import torch
from mmpretrain import get_model

model = get_model('colormae_vit-base-p16_8xb512-amp-coslr-300e_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

## Pretraining Instructions

We use mmpretrain for pretraining the models similar to [MAE](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mae). Please refer here for the instructions: [PRETRAIN.md](PRETRAIN.md).

## Finetuning Instructions
We evaluate transfer learning performance using our pre-trained ColorMAE models on different datasets and downstream tasks including: Image Classification, Semantic Segmentation, and Object Detection. Please refer here for the instructions: [FINETUNE.md](benchmarks/FINETUNE.md).

## Acknowledgments

- Our code is based on the MAE implementation of the mmpretrain project: https://github.com/open-mmlab/mmpretrain/tree/main/configs/mae. We thank all contributors from [MMPreTrain](https://github.com/open-mmlab/mmpretrain), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), and [MMDetection](https://github.com/open-mmlab/mmdetection).
- This work was supported by the KAUST Center of Excellence on **GenAI** under award number **5940**.

<!-- How to Cite -->
## How to cite

If you use our code or models from this project in your research, please cite our work as follows:

```Latex
@article{hinojosa2024colormae,
  title={ColorMAE: Exploring data-independent masking strategies in Masked AutoEncoders},
  author={Hinojosa, Carlos and Liu, Shuming and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2407.13036},
  url={https://arxiv.org/pdf/2407.13036}
  year={2024}
}
```

## Troubleshooting

### CuDNN Warning

If you encounter the following warning at the beginning of pretraining:
```text
UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggered internally at /opt/conda/conda-bld/pytorch_1682343995026/work/aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv2d(input, weight, bias, self.stride,
```
**Solution:** This warning indicates a missing or incorrectly linked nvrtc.so library in your environment. To resolve this issue, create a symbolic link to the appropriate libnvrtc.so file. Follow these steps:
1. Navigate to the library directory of your virtual environment:
```bash
cd venv/lib/  # Adjust the path if your environment is located elsewhere
```
2. Create a symbolic link to libnvrtc.so.11.8.89:
```bash
ln -sfn libnvrtc.so.11.8.89 libnvrtc.so
```