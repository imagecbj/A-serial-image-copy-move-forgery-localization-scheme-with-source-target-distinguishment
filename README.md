# A serial image copy-move forgery localization scheme with source/target distinguishment

#### Overview
In this paper, we improve the existing parallel deep neural network (DNN) scheme (BusterNet) for image copy-move forgery localization with source/target distinguishment. To do so, it is based on two branches, i. e., Simi-Det and Mani-Det, but suffers of two main drawbacks: (a) It should ensure that both branches locate regions correctly; (b) Simi-Det branch only extracts single-level and low-resolution features by VGG16 with four pooling layers. To be sure of source and target regions identification, we introduce two sub-networks, that are constructed in a serial way, and named: copy-move similarity detection network (CMSDNet), and source/target regions distinguishment network (STRDNet).
[Chen B, Tan W, Coatrieux G, et al. A serial image copy-move forgery localization scheme with source/target distinguishment[J]. IEEE Transactions on Multimedia, 2020.](https://ieeexplore.ieee.org/abstract/document/9207851/)

#### Prerequisites
- Linux
- NVIDIA GPU+CUDA CuDNN 
- Install TensorFlow and dependencies

#### Training and Test Details
For training and testing details，please refer to the proposed paper.

#### Using

1.  Place the test image in the root directory and name it as 'test.png'
2.  Run CMSDNetTest.py, then you will get the result of CMSDNet
3.  Run STRDNetTest.py, then you will get the result of STRDNet

#### Related Works

1.  Y. Wu, W. Abd-Almageed, and P. Natarajan, ‘‘BusterNet: Detecting
    copy-move image forgery with source/target localization,’’ in Proc. Eur.
    Conf. Comput. Vis. (ECCV)., pp. 168–184, 2018.
