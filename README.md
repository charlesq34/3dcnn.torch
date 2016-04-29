# Volumetric CNN (Convolutional Neural Networks) for Object Classification on 3D Data

[Teaser picture + MATLAB code to extract feature]

### Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/1604.03265). Our paper will also appear as a CVPR 2016 spotlight (please refer to the arXiv one for most up-to-date results). In this repository, we release code, data for training Volumetric CNNs for object classification on 3D data. We also provide trained models and a MATLAB interface to extract 3D features from a binary volume.

### Contents
1. [Train Volumetric CNNs with Torch](#train-volumetric-cnns-with-torch)
2. [Caffe Models and Reference Results](#caffe-models-and-reference-results)

### Train Volumetric CNNs with Torch

** Prerequisites **

<a href="" target="_blank">Install Torch with CUDNN</a>

To train a model to classify 3D object:

    <pre>
    th train.lua
    </pre>

Voxelizations of ModelNet40 models in HDF5 files will be automatically downloaded (around 500MB). `modelnet40_60x` include azimuth and elevation rotation augmented occupancy grids. `modelnet40_12x` is of azimuth rotation augmentation only. `modelnet40_20x_stack` is used for multi-orientation training.

### Caffe Models and Reference Results

Caffe prototxt files of models reported in the paper have been included in the `caffe_models` directory.

### License
Our code and models are released under MIT License (see LICENSE file for details).

### Citation
If you find our work useful in your research, please consider citing:

    @article{qi2016volumetric,
        title={Volumetric and Multi-View CNNs for Object Classification on 3D Data},
        author={Qi, Charles R and Su, Hao and Niessner, Matthias and Dai, Angela and Yan, Mengyuan and Guibas, Leonidas J},
        journal={arXiv preprint arXiv:1604.03265},
        year={2016}
    }


