# A serial image copy-move forgery localization scheme with source/target distinguishment

#### Overview
In this paper, we improve the existing parallel deep neural network (DNN) scheme (BusterNet) for image copy-move forgery localization with source/target distinguishment. To do so, it is based on two branches, i. e., Simi-Det and Mani-Det, but suffers of two main drawbacks: (a) It should ensure that both branches locate regions correctly; (b) Simi-Det branch only extracts single-level and low-resolution features by VGG16 with four pooling layers. To be sure of source and target regions identification, we introduce two sub-networks, that are constructed in a serial way, and named: copy-move similarity detection network (CMSDNet), and source/target regions distinguishment network (STRDNet).

[Chen B, Tan W, Coatrieux G, et al. A serial image copy-move forgery localization scheme with source/target distinguishment[J]. IEEE Transactions on Multimedia, 2020.](https://ieeexplore.ieee.org/abstract/document/9207851/)

#### Prerequisites
- Linux
- NVIDIA GPU+CUDA CuDNN 
- Install TensorFlow and dependencies

#### Training and Test Details
- Training: Wu et al. [1] created a new synthetic dataset with 100,100 samples. Similar to [1], the synthetic dataset is split into training, validation, and testing sets with a ratio of 8:1:1. More specifically, the parameter initialization of all layers uses the default function of Keras. We find that CMSDNet converges after approximately 15 epochs of training. In the first 10 epochs, we use a minibatch gradient descent optimizer with momentum 0.9 and set an initial learning rate of 1.0e-3 and a minibatch size of 16. When validation loss reaches plateaus after 10 epochs, we reduce the learning rate and set it to 1.0e-4 for 5 epochs more. Regarding the STRDNet, it converges after approximately 10 epochs of training. Some optimizer settings are the same as CMSDNet, while the learning rate is always kept at 1.0e-3 and the minibatch size is set to 64.
- Testing: To test the generalization ability of our algorithm, three standard datasets, i.e., CASIA v2.0, CoMoFoD and COVERAGE, are considered to evaluate the performance of the trained model obtained from the synthetic dataset of Wu et al. 

#### Using

1.  Place the test image in the root directory and name it as 'test.png'
2.  Run CMSDNetTest.py, then you will get the result of CMSDNet
3.  Run STRDNetTest.py, then you will get the result of STRDNet

#### Related Works

1.  Y. Wu, W. Abd-Almageed, and P. Natarajan, ‘‘BusterNet: Detecting
    copy-move image forgery with source/target localization,’’ in Proc. Eur.
    Conf. Comput. Vis. (ECCV)., pp. 168–184, 2018.
