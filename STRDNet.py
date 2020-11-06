from keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D, Layer, Input,
                          Activation, BatchNormalization)
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import numpy as np
import keras
import tensorflow as tf
import warnings
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")


class Preprocess(Layer):
    def call(self, x, mask=None):
        xout = preprocess_input(x, mode="tf")
        # rgb2gray scale
        xout = tf.image.rgb_to_grayscale(xout)
        return xout

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128, 128, 1)


def standard_conv_bn_re(x, out_kernals, ksize, name, stride):
    conv = Conv2D(
        filters=out_kernals,
        kernel_size=ksize,
        activation="linear",
        padding="same",
        strides=(stride, stride),
        name=name + "_conv",
    )(x)
    xn_norm = BatchNormalization(name=name + "_bn")(conv)
    xn = Activation("tanh", name=name + "_re")(xn_norm)
    return xn


def target_source_mani(img_shape=(128, 128, 1), name="simiDet"):
    img_input = Input(shape=img_shape, name=name + "_in")
    bname = name + "_cnn"
    x1 = Conv2D(
        filters=3,
        kernel_size=(5, 5),
        activation="linear",
        padding="same",
        strides=(1, 1),
        use_bias=False,
        name=name + "_c1",
    )(img_input)
    x2 = standard_conv_bn_re(x1, 96, (7, 7), bname + "_c2", 2)
    x2_p = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                        name=bname + "_x2_p")(x2)
    x3 = standard_conv_bn_re(x2_p, 64, (5, 5), bname + "_c3", 1)
    x3_p = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                        name=bname + "_x3_p")(x3)
    x4 = standard_conv_bn_re(x3_p, 64, (5, 5), bname + "_c4", 1)
    x4_p = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                        name=bname + "_x4_p")(x4)
    x5 = standard_conv_bn_re(x4_p, 128, (1, 1), bname + "_c5", 1)
    x5_p = AveragePooling2D(pool_size=(3, 3),
                            strides=(2, 2),
                            name=bname + "_x5_p")(x5)
    # FC
    bname = name + "_fc"
    fc_pre = Flatten()(x5_p)
    fc1 = Dense(units=200, activation="tanh")(fc_pre)
    fc1 = Dropout(rate=0.75)(fc1)
    fc2 = Dense(units=200, activation="tanh")(fc1)
    fc2 = Dropout(rate=0.75)(fc2)
    fc_fin = Dense(units=2, activation="softmax")(fc2)
    model = Model(inputs=img_input, outputs=fc_fin, name=name)
    return model


def creat_test_model(model=None):
    # ts_mani
    ts_mani = target_source_mani()
    ts_mani_Det = Model(inputs=ts_mani.inputs,
                        outputs=ts_mani.layers[-1].output,
                        name="ts_mani_Featex")
    img_raw = Input(shape=(None, None, 3), name="image_in")
    img_in = Preprocess(name="preprocess")(img_raw)
    mani_feat = ts_mani_Det(img_in)
    train_model = Model(inputs=img_raw, outputs=mani_feat, name="ts_mani_Det")
    if model is not None:
        try:
            train_model.load_weights(model, by_name=True)
            print("INFO: successfully load pretrained weights from {}".format(
                model))
        except Exception as e:
            print(
                "INFO: fail to load pretrained weights from {} for reason: {}".
                format(model, e))
    return train_model


if __name__ == "__main__":
    tf.app.run()