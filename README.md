# 3dcnn.torch
Volumetric CNN (Convolutional Neural Networks) for Object Classification on 3D Data, with Torch implementation.


### Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/1604.03265). Our paper will also appear as a CVPR 2016 spotlight (please refer to the arXiv one for most up-to-date results). In this repository, we release code, data for training Volumetric CNNs for object classification on 3D data (binary volume).


### Installation

Install <a href="http://torch.ch/docs/getting-started.html" target="_blank">Torch7</a>.

Note that cuDNN and GPU are required for VolumetricBatchNormalization layer. 
You also need to install a few torch packages (if you haven't done so) including `cudnn.troch`, `cunn`, `hdf5` and `xlua`.


### Usage
To train a model to classify 3D object:

    th train.lua

Voxelizations of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files will be automatically downloaded (633MB). `modelnet40_60x` include azimuth and elevation rotation augmented occupancy grids. `modelnet40_12x` is of azimuth rotation augmentation only. `modelnet40_20x_stack` is used for multi-orientation training. There are also text files in each folder specifying the sequence of CAD models in h5 files.

To see options for training just type:

    th train.lua -h

After the above training, which trains a 3D CNN classifying object based on single input, you can then train a multi-orientation 3D CNN by initializing it with the pretrained network that takes single orientation input:

    th train_mo.lua --model <network_name> --model_param_file <model_param_filepath>

You need to specify at which layer to max-pool the feature in the network, it can be either set in command line by `--pool_layer_idx <layer_idx>` or set it interactively when after the program starts and prints the network layers and indices. For `3dnin_fc`, `--pool_layer_idx 27` which will max pool the outputs of the last convolutional layer from multiple orientations of a volume.

Below are the classification accuracis we got on ModelNet40 test data.

|   Model       | Single-Orientation    | Multi-Orientation  |
| ------------- |:---------------------:|:------------------:|
| voxnet        | 86.2% (on 12x data)   | -                  | 
| 3dnin fc      | 88.8%                 | 90.3%              |
| subvol sup    | 88.8%                 | 90.1%              |
| ani probing   | 87.5%                 | 90.0%              |

<b>Note 1:</b> Compared with the cvpr paper, we have included batch normalization and drop layers after all the convolutional layers (in caffe prototxt there are only dropouts after fc layers and no batch normalization is used). We also add small translation jittering to augment the data on the fly (following what <a href="https://github.com/dimatura/voxnet" target="_blank">voxnet</a> has done). Besides the two models proposed in the paper (subvolume supervised network and anisotropic probing network), we have also found a variation of the base network used in `subvolume_sup`, which we call `3dnin_fc` here. `3dnin_fc` has a relatively simple architecture (3 mlpconv layers + 2 fc layers) and performs also very well, so we have set it as the default architecture to use in this repository.

<b>Note 2:</b> Numbers reported in the table above are average instance accuracy on the whole ModelNet40 test set containing 2468 CAD models from 40 categories.This different from what is on the modelnet website, which is average class accuracy on either a subset of the test set or the full test set. For direct comparison under average class accuracy metric, please refer to our [paper](https://arxiv.org/abs/1604.03265).


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
