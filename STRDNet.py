from keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D, Layer, Input,
                          Activation, BatchNormalization)
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import (
    ModelCheckpoint,
    Callback,
)
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import keras
import tensorflow as tf
import warnings
import os
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size1", "64", "batch size for training")
tf.flags.DEFINE_integer("NUM_EPOCHS_TRAIN1", "100", "NUM_EPOCHS for training")
tf.flags.DEFINE_string("weights_path1",
                       "logs/hdf5/weights_target_source/target_source_Bayar_for_100k",
                       "path to save model weights")
tf.flags.DEFINE_string("model_weights1", "model/weights_st_no_data_augmentation.hdf5",
                       "saved weights")
tf.flags.DEFINE_string("model_name", "weights_st_no_data_augmentation.hdf5",
                       "load weights to test model performance")
tf.flags.DEFINE_float("learning_rate1", "1e-5", "Learning rate for Optimizer")
tf.flags.DEFINE_float("dp_rate1", "0.75",
                      "dp rate for NN")
tf.flags.DEFINE_float(
    "weight_decay1", "1e-3",
    "regularizer rate for weights")

IMAGE_SIZE = 128
IMAGE_CHANNELS = 1
W_regularizer = l2(FLAGS.weight_decay1)


class Preprocess(Layer):
    def call(self, x, mask=None):
        # substract channel means if necessary
        xout = preprocess_input(x, mode="tf")
        # rgb2gray scale
        xout = tf.image.rgb_to_grayscale(xout)
        return xout

    def compute_output_shape(self, input_shape):
        return (input_shape[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)


def standard_conv_bn_re(x, out_kernals, ksize, name, stride):
    conv = Conv2D(
        filters=out_kernals,
        kernel_size=ksize,
        activation="linear",
        padding="same",
        strides=(stride, stride),
        kernel_regularizer=W_regularizer,
        name=name + "_conv",
    )(x)
    xn_norm = BatchNormalization(name=name + "_bn")(conv)
    xn = Activation("tanh", name=name + "_re")(xn_norm)
    return xn


def target_source_mani(img_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
                       name="simiDet"):
    # input
    img_input = Input(shape=img_shape, name=name + "_in")
    bname = name + "_cnn"
    # high filters
    filter1 = [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]
    filter1 = np.asarray(filter1, dtype=float) / 12
    filters = [[filter1], [filter1], [filter1]]
    filters = np.einsum('klij->ijlk', filters)
    # CNN
    # x1 = Conv2D(
    #     filters=3,
    #     kernel_size=(5, 5),
    #     activation="linear",
    #     padding="same",
    #     strides=(1, 1),
    #     use_bias=False,
    #     kernel_regularizer=W_regularizer,
    #     kernel_initializer=keras.initializers.Constant(value=filters),
    #     trainable=False,
    #     name=name + "_c1",
    # )(img_input)
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
    fc1 = Dropout(rate=FLAGS.dp_rate1)(fc1)
    fc2 = Dense(units=200, activation="tanh")(fc1)
    fc2 = Dropout(rate=FLAGS.dp_rate1)(fc2)
    fc_fin = Dense(units=2, activation="softmax")(fc2)
    model = Model(inputs=img_input, outputs=fc_fin, name=name)
    return model


def main(argv=None):
    # dataset for train and valid
    f_train = h5py.File('data_zoo/syntic_for_train_100k.hd5')
    x_train, y_train = f_train["X"], f_train["Y"]
    y_train = to_categorical(y_train, num_classes=2)
    f_valid = h5py.File('data_zoo/syntic_for_valid_20k.hd5')
    x_valid, y_valid = f_valid["X"], f_valid["Y"]
    y_valid = to_categorical(y_valid, num_classes=2)

    # define train model-ts_mani
    ts_mani = target_source_mani()
    ts_mani_Det = Model(inputs=ts_mani.inputs,
                        outputs=ts_mani.layers[-1].output,
                        name="ts_mani_Featex")

    img_raw = Input(shape=(None, None, 3), name="image_in")
    img_in = Preprocess(name="preprocess")(img_raw)
    mani_feat = ts_mani_Det(img_in)
    train_model = Model(inputs=img_raw, outputs=mani_feat, name="ts_mani_Det")

    opt = keras.optimizers.SGD(lr=FLAGS.learning_rate1, momentum=0.95)
    train_model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["acc"],
    )
    train_model.summary()
    # define save_model callback
    if not os.path.exists(FLAGS.weights_path1 + "/model"):
        os.makedirs(FLAGS.weights_path1 + "/model")

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(FLAGS.weights_path1,
                              "weights_{epoch:02d}-{val_acc:.3f}.hdf5"),
        save_weights_only=True,
        verbose=1,
        save_best_only=False,
        mode="min",
    )
    path = os.path.join(FLAGS.weights_path1, FLAGS.model_weights1)
    if os.path.exists(path):
        train_model.load_weights(path, by_name=True)
        print("INFO: successfully restore train_weights from: {}".format(path))
    else:
        print("INFO: train_weights: {} not exist".format(path))

    # for layer in train_model.layers:
    #     print(layer)
    #     for weight in layer.weights:
    #         print(weight)
    train_model.summary()

    train_model.fit(
        x_train,
        y_train,
        batch_size=FLAGS.batch_size1,
        epochs=FLAGS.NUM_EPOCHS_TRAIN1,
        callbacks=[checkpointer],
        validation_data=(x_valid, y_valid),
        shuffle="batch",
        # shuffle=True,
    )
    # save Model
    train_model.save_weights(
        os.path.join(FLAGS.weights_path1, "train_all_weights.h5"))


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