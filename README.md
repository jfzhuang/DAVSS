# Video Semantic Segmentation with Distortion-Aware Feature Correction
This repository is the official implementation of "Video Semantic Segmentation with Distortion-Aware Feature Correction" (accepted by IEEE Transactions on Circuits and Systems for Video Technology(TCSVT) 2020). It is designed for efficient video semantic segmentation task.

[Paper](https://arxiv.org/abs/2006.10380) | [Project Page](https://jfzhuang.github.io/DAVSS.github.io/) | [YouTube]() | [BibeTex](#citation)

<img src="./gif/demo.gif" width="860"/>

## Install & Requirements
The code has been tested on pytorch=1.5.0 and python3.7. Please refer to `requirements.txt` for detailed information.

**To Install python packages**
```
pip install -r requirements.txt
```

**To Install resampled 2d modules**
```
cd $DAVSS_ROOT/lib/model/resample2d_package
python setup.py build
```

## Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid//) datasets.

Your directory tree should be look like this:
````bash
$DAVSS_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
├── camvid
│   ├── label
│   │   ├── segmentation annotations
│   └── video_image
│       ├── 0001TP
│           ├── decoded images from video clips
│       ├── 0006R0
│       └── 0016E5
│       └── Seq05VD
````

## Experiment preparation
### Environment Setting
```
export PYTHONPATH=$PYTHONPATH:$DAVSS_ROOT
cd $DAVSS_ROOT
```

### Download pretrained model
We provide pretrained deeplabv3+ and flownet models on Cityscapes and CamVid datasets. You can download from [OneDrive](https://www.dropbox.com/sh/4eqce1lj75ks76v/AACjw8e5yVpeLEf1CDs4SsXea?dl=0)/[BaiduYun(Access Code:r4cd)](https://pan.baidu.com/s/1R2dfkFd_1KumrVpVY2_2iw). Please place pretrained models in ./saved_model/pretrained.

## Train and test
Please specify the script file.

For example, train our proposed method on Cityscapes on 4 GPUs:
````bash
# training DMNet
bash ./exp/dmnet_cityscapes/script/train.sh
# training the entire frameowrk
bash ./exp/spatial_correction_cityscapes/script/train.sh
````

For example, test our proposed method on Cityscapes validation set with PDA evaluation:
````bash
bash ./exp/spatial_correction_cityscapes/script/test_PDA.sh
````

For example, visualize our proposed method on Cityscapes validation set:
````bash
bash ./exp/spatial_correction_cityscapes/script/show.sh
````
Obtained results are saved in ./result/spatial_correction_cityscapes.

Conducting experiments on the CamVid dataset should follow the above procedure similarly.

## Trained model
We provide trained model on Cityscapes and CamVid datasets. Please download models from:
| model | Link |
| :--: | :--: |
| dmnet_camvid | [OneDrive](https://www.dropbox.com/sh/ahrybyqrhhtgo68/AAATwdQUYum-M7Pwp67vrr2ua?dl=0)/[BaiduYun(Access Code:iy69)](https://pan.baidu.com/s/1-hdhe-CPf3sMPWVFzlUebQ) |
| spatial_correction_camvid | [OneDrive](https://www.dropbox.com/sh/sb7s8t3epp68pxm/AADm7xsXywEDxb_kjKUHQ5AUa?dl=0)/[BaiduYun(Access Code:tukd)](https://pan.baidu.com/s/1AL_nZUEQ1p9-fJqz_0yOpg) |
| dmnet_cityscapes | [OneDrive](https://www.dropbox.com/sh/25v31t5s4tzk804/AAC9TExLemqrRfh7B-1HyqG_a?dl=0)/[BaiduYun(Access Code:rc7u)](https://pan.baidu.com/s/1rQPlL3v4-tS6ECBmslMP7g) |
| spatial_correction_cityscapes | [OneDrive](https://www.dropbox.com/sh/9f27itrt2op9zri/AADAgS7IEJGbZAkXseik7FMQa?dl=0)/[BaiduYun(Access Code:5gem)](https://pan.baidu.com/s/1JRH-3Kz893OlcjOnS_ZXTQ) |

## Citation
```
@article{zhuang2020video,
  title={Video Semantic Segmentation with Distortion-Aware Feature Correction},
  author={Zhuang, Jiafan and Wang, Zilei and Wang, Bingke},
  journal={arXiv preprint arXiv:2006.10380},
  year={2020}
}
```
