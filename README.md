# CMPT
This is the PyTorch implementation repository of our [IJCV 2021] paper: Cross-Modal Pyramid Translation for RGB-D Scene Recognition

## Description
We propose a general and flexible multi-modal learning framework, to unify the tasks of modality specific recognition and cross-modal image translation with a shared representation, thus improving its generalization power. A preliminary version of this work was presented in the conference paper: TRecgNet (https://github.com/ownstyledu/Translate-to-Recognize-Networks). 

## Development Environment
+ Python 3.6 or higher
+ PyTorch 1.4 or higher
+ Torchvision
+ TensorboardX

## Datasets
We provide links(https://pan.baidu.com/s/1LZIF1hlT3k0oX76Ttp660w, extraction code: g5vp) to download SUN RGB-D data in ImageFolder format and depth data has been encoded using HHA format. The code runs with the concatenated version of this data.

## Usage
One can easily config all the hyper-parameters using the files in the *config* package.
An example to run the Depth TRecgNet (depth to RGB) with *multiscale* image translation is shown as follows (0 refers to the gpu idx):

`bash train.sh 0 sunrgbd arch=resnet18 content_arch=resnet18 multi_scale=True direction=BtoA model=trecg no_trans=False loss_types=CLS,PERCEPTUAL,AUX_CLS`

## Citation
Please cite the following paper if you feel this repository useful.
```
@article{du2021cross,
  title={Cross-Modal Pyramid Translation for RGB-D Scene Recognition},
  author={Du, Dapeng and Wang, Limin and Li, Zhaoyang and Wu, Gangshan},
  journal={International Journal of Computer Vision},
  volume={129},
  number={8},
  pages={2309--2327},
  year={2021},
  publisher={Springer}
}

@inproceedings{du2019translate,
  title={Translate-to-Recognize Networks for RGB-D Scene Recognition},
  author={Du, Dapeng and Wang, Limin and Wang, Huiling and Zhao, Kai and Wu, Gangshan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11836--11845},
  year={2019}
}

```
