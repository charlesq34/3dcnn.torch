# Volumetric CNN (Convolutional Neural Networks) for Object Classification on 3D Data

### Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/1604.03265). Our paper will also appear as a CVPR 2016 spotlight (please refer to the arXiv one for most up-to-date results). In this repository, we release code, data for training Volumetric CNNs for object classification on 3D data. We also provide trained models and a MATLAB interface to extract 3D features from a binary volume.

### Contents
1. [Train Volumetric CNNs with Torch](#train-volumetric-cnns-with-torch)
2. [Caffe Models and Reference Results](#caffe-models-and-reference-results)

### Train Volumetric CNNs with Torch

** Prerequisites **

<a href="http://torch.ch/docs/getting-started.html" target="_blank">Install Torch</a>. Note that cuDNN and GPU are required for VolumetricBatchNormalization layer. You also need to install a few torch packages including `cudnn.troch`, `cunn`, `hdf5` and `xlua`.

To train a model to classify 3D object:

    th train.lua

Voxelizations of ModelNet40 models in HDF5 files will be automatically downloaded (633MB). `modelnet40_60x` include azimuth and elevation rotation augmented occupancy grids. `modelnet40_12x` is of azimuth rotation augmentation only. `modelnet40_20x_stack` is used for multi-orientation training. There are also text files in each folder specifying the sequence of CAD models in h5 files.

To see options for training just type:

    th train.lua -h

After the above training, which trains a 3D CNN classifying object based on single input, you can then train a multi-orientation 3D CNN by initializing it with the pretrained network that takes single orientation input:

    th train_mo.lua --model <network_name> --model_param_file <model_param_filepath>


Below are the classification accuracis we got on ModelNet40 test data:


|   Model       | Single-Orientation    | Multi-Orientation  |
| ------------- |:---------------------:|:------------------:|
| voxnet        | 86.2% (on 12x data)   | -                  | 
| 3dnin fc      | 88.8%                 | 90.3%              |
| subvol sup    | 88.8%                 | 90.1%              |
| ani probing   | 87.5%                 | 90.0%              |

Note that compared with the cvpr paper, we have included batch normalization and drop layers after all the convolutional layers (in caffe prototxt there are only dropouts after fc layers and no batch normalization is used). We also add small translation jittering to augment the data on the fly (following what <a href="https://github.com/dimatura/voxnet" target="_blank">voxnet</a> has done). Besides the two models proposed in the paper (subvolume supervised network and anisotropic probing network), we have also found a variation of the base network used in `subvolume_sup`, which we call `3dnin_fc` here. `3dnin_fc` has a relatively simple architecture (3 mlpconv layers + 2 fc layers) and performs also very well, so we have set it as the default architecture to use in this repository.

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

### Acknowledgement
Torch implementation in this repository is based on the code from <a href="https://github.com/szagoruyko/cifar.torch" target="_blank">cifar.torch</a>, which is a clean and nice GitHub repo on CIFAR image classification using Torch.

### TODO

Add matlab interface to extract 3d feature.
