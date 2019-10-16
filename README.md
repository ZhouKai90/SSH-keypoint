# SSH-keypoint
### SSH-keypoint: Single Stage Headless Face Detector with keypoint

This repository includes the training and evaluating code for the SSH face detector introduced in the [ICCV 2017](https://arxiv.org/abs/1708.03979) paper. Moreover, we achieved 5 keypoints prediction in this repository. 

#### result
![](https://github.com/ZhouKai90/SSH-keypoint/blob/master/images/image1.jpg)
![](https://github.com/ZhouKai90/SSH-keypoint/blob/master/images/image2.jpg)
![](https://github.com/ZhouKai90/SSH-keypoint/blob/master/images/image3.jpg)

#### evaluate
![](https://github.com/ZhouKai90/SSH-keypoint/blob/master/evaluation.jpg)
#### Train

This repository train and evaluate on the Mxnet, make sure your environment can compile Mxnet with cuDNN successfully.

1. Clone this repository

   `https://github.com/ZhouKai90/SSH-keypoint.git`

2. compile the rcnn cpython code

   goto your local repository path and run`make`

3. We train this repository base on widerface dataset. You need download widerface dataset with keypoint annotations into `data`.Modify`tools/anno_widerface_to_pascal.py` to your own parameters.

   `cd tools`

   `python anno_widerface_to_pascal.py`

4. Modify`utils/config.py`to your own for the training parameters.

5. run `python train/train_ssh.py` for training.

#### Test
run `python demo/test.py`
   

