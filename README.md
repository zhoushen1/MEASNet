# <p align=center> Multi-Expert Adaptive Selection: Task-Balancing for All-in-One Image Restoration</p>

## Continuously updating!  
Note: Still under review, this code repository is not yet fully complete.

## 1.Quick Start

### Install
This repository is built in PyTorch 1.12.0 and Python 3.8
Follow these intructions
1. Clone our repository
```
git clone  https://github.com/zhoushen1/MEASNet
cd MEASNet
```
2. Create conda environment
The Conda environment used can be recreated using the ```env.yml``` file
```
conda env create -f env.yml
```
### Datasets
Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/1e62XGdi5c6IbvkZ70LFq0KLRhFvih7US/view?usp=sharing),[BSD68](https://github.com/clausmichele/CBSD68-dataset/tree/master/CBSD68/original)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: Train[ RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2), Test [SOTS-Outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

Deblur: [GoPro](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view?usp=sharing)

Enhance: [LOL-V1](https://daooshee.github.io/BMVC2018website/)

The training data should be placed in ``` data/Train/{task_name}```.

The testing data should be placed in ```data/test/{task_name}```. 

## 2.Training
After preparing the training data in ```data/``` directory, use 
```
python train.py
```
## 3.Testing

After preparing the testing data in ```test/``` directory, use
```
python test.py
```

## Results
You can download visual results from (Link：https://pan.baidu.com/s/1GHmqP9himlZ_yo9h2AYCCQ?pwd=o2kp code：o2kp)
# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.
    Zhou Shen
    Faculty of Information Engineering and Automation Kunming University of Science and Technology                                                           
    Email: zhoushennn@163.com

## Thanks
- See More Details: Efficient Image Super-Resolution by Experts Mining, ICML 2024. [Paper](https://arxiv.org/abs/2402.03412) | [Code](https://github.com/eduardzamfir/seemoredetails)
