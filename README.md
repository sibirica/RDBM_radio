# Project

# Residual Diffusion Bridge Model for Image Restoration

<em>Hebaixu Wang, Jing Zhang, Haoyang Chen, Haonan Guo, Di Wang, Jiayi Ma and Bo Du</em>.

[Paper](https://arxiv.org/abs/2510.23116) |  [Github Code](https://github.com/MiliLab/RDBM)

## Abstract

Diffusion bridge models establish probabilistic paths between arbitrary paired distributions and exhibit great potential for universal image restoration. Most existing methods merely treat them as simple variants of stochastic interpolants, lacking a unified analytical perspective. Besides, they indiscriminately reconstruct images through global noise injection and removal, inevitably distorting undegraded regions due to imperfect reconstruction. To address these challenges, we propose the {R}esidual {D}iffusion {B}ridge {M}odel (RDBM). Specifically, we theoretically reformulate the stochastic differential equations of generalized diffusion bridge and derive the analytical formulas of its forward and reverse processes. Crucially, we leverage the residuals from given distributions to modulate the noise injection and removal, enabling adaptive restoration of degraded regions while preserving intact others. Additionally, we unravel the fundamental mathematical essence of existing bridge models, all of which are special cases of RDBM and empirically demonstrate the optimality of our proposed models. Extensive experiments are conducted to demonstrate the state-of-the-art performance of our method both qualitatively and quantitatively across diverse image restoration tasks.

## Introducation

<img src="./assets/intro.png" width="100%">

## Overview

<img src="./assets/method.png" width="100%">

## Stochastic Trajectories

<img src="./assets/sde.png" width="100%">

## Visualization

<img src="./assets/visualization.png" width="100%">

<img src="./assets/application.png" width="100%">

## Datasets Information

| Task                     | Dataset                        | Synthetic/Real      | Download Links |
|--------------------------|--------------------------------|---------------------|----------------|
| **Deraining**            | DID                            | Synthetic           | [URL](https://github.com/hezhangsprinter/DID-MDN)                                                                                       |
|                          | DeRaindrop                     | Real                | [URL](https://github.com/rui1996/DeRaindrop)                                                                                            |
|                          | Rain13K                        | Synthetic           | [URL](https://github.com/kuijiang94/MSPFN)                                                                                              |
|                          | Rain_100H                      | Synthetic           | [URL](https://github.com/kuijiang94/MSPFN)                                                                                              |
|                          | Rain_100L                      | Synthetic           | [URL](https://github.com/kuijiang94/MSPFN)                                                                                              | 
|                          | GT-Rain                        | Real                | [URL](https://github.com/UCLA-VMG/GT-RAIN)                                                                                              | 
|                          | RealRain-1k                    | Real                | [URL](https://github.com/hiker-lw/RealRain-1k)                                                                                          | 
| **Low-light Enhancement**| LOL                            | Real                | [URL](https://github.com/weichen582/RetinexNet?tab=readme-ov-file)                                                                      |
|                          | MEF                            | Real                | [URL](https://ieeexplore.ieee.org/abstract/document/7120119)                                                                            |
|                          | VE-LOL-L                       | Synthetic/Real      | [URL](https://flyywh.github.io/IJCV2021LowLight_VELOL/)                                                                                 | 
|                          | NPE                            | Real                | [URL](https://ieeexplore.ieee.org/abstract/document/6512558)                                                                            | 
| **Desnowing**            | CSD                            | Synthetic           | [URL](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet)                                                         | 
|                          | Snow100K-Real                  | Real                | [URL](https://sites.google.com/view/yunfuliu/desnownet)                                                                                 |
| **Dehazing**             | SOTS                           | Synthetic           | [URL](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=3D0)                                                | 
|                          | ITS_v2                         | Synthetic           | [URL](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=3D0)                                                | 
|                          | D-HAZY                         | Synthetic           | [URL](https://www.cvmart.net/dataSets/detail/559?channel_id=op10&utm_source=cvmartmp&utm_campaign=datasets&utm_medium=article)          |
|                          | NH-HAZE                        | Real                | [URL](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)                                                                              |
|                          | Dense-Haze                     | Real                | [URL](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/)                                                                           |
|                          | NHRW                           | Real                | [URL](https://github.com/chaimi2013/3R)                                                                                                 | 
| **Deblur**               | GoPro                          | Synthetic           | [URL](https://github.com/SeungjunNah/DeepDeblur-PyTorch)                                                                                | 
|                          | RealBlur                       | Real                | [URL](https://github.com/rimchang/RealBlur)      
 
### Contributor

Baixuzx7 @ wanghebaixu@gmail.com

### Citation
 
```
@inproceedings{wang2026residual,
  title={Residual diffusion bridge model for image restoration},
  author={Wang, Hebaixu and Zhang, Jing and Chen, Haoyang and Guo, Haonan and Wang, Di and Ma, Jiayi and Du, Bo},
  booktitle={Proceedings of the Conference on Computer Vision and Pattern Recognition},
  pages={8375--8386},
  year={2026}
}
```

### Copyright statement

The project is signed under the MIT license, see the [LICENSE.md](https://github.com/MiliLab/RDBM/LICENSE.md)

