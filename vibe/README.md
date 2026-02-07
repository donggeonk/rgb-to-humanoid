# VIBE

This is a minimal, inference only implementation of VIBE. 

## Weights

Please download the checkpoint from: [this link](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/vibe/resnet50_vibe_pw3d-2e05a122_20211201.pth?versionId=CAEQHhiBgMCNvonS6xciIGEyOGM1M2M0ZTdiMDQ4NTc4NDI1MjBmYzgyMjUwMWI2)
## Usage

To use on a video,  run:

```shell
python demo_video.py --video input.mp4 --checkpoint resnet50_vibe_pw3d-2e05a122_20211201.pth --visualize
```


Also, you can run on the webcam:

```shell
python demo_webcam.py --checkpoint resnet50_vibe_pw3d-2e05a122_20211201.pth --visualize
```
