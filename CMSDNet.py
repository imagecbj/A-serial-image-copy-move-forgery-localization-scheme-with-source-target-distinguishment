import tensorflow as tf
import keras
import os
import numpy as np
import warnings
import CreatImagesList as creat_dataset
from batch_generator import read_images, read_validations
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Layer, Input, Lambda, Add, Multiply
from keras.layers.core import Reshape, Dense, Dropout
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import (
    ModelCheckpoint,
    Callback,
    ReduceLROnPlateau,
    EarlyStopping,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_integer("NUM_EPOCHS_TRAIN", "20", "NUM_EPOCHS for training")
tf.flags.DEFINE_string("weights_path",
                       "logs/hdf5/weights_perason_fm_115/run2",
                       "path to save model weights")
tf.flags.DEFINE_string("model_weights",
                       "model/f1_CoMoFoD-0.5179_CASIA-0.5514.hdf5",
                       "saved weights")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Optimizer")
tf.flags.DEFINE_float("decay_rate", "0.5", "decay rate for learning rate")
tf.flags.DEFINE_float(
    "weight_decay", "1e-8",
    "regularizer rate for weights")  # 1e-4 (0.46..); 2.5e-5 ?/5e-5 (0.49..);
tf.flags.DEFINE_float("dp_rate", "0.5", "dp rate for NN")  # 0.5

IMAGE_SIZE = 256
CHANNELS = 3
W_regularizer = l2(FLAGS.weight_decay)


def iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), 1), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), 1), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def f1_avg(y_true, y_pred):
    def recall(y_true, y_pred):
        y_true = K.flatten(K.argmax(y_true))
        y_pred = K.flatten(K.argmax(y_pred))
        true_positives = K.sum(y_true * y_pred)
        true_positives = K.cast(true_positives, "float32")
        possible_positives = K.sum(y_true)
        possible_positives = K.cast(possible_positives, "float32")
        recall_item = true_positives / possible_positives
        return recall_item

    def precision(y_true, y_pred):
        y_true = K.flatten(K.argmax(y_true))
        y_pred = K.flatten(K.argmax(y_pred))
        true_positives = K.sum(y_true * y_pred)
        true_positives = K.cast(true_positives, "float32")
        predicted_positives = K.clip(
            K.cast(K.sum(y_pred), "float32"), K.epsilon(), None)
        precision_item = true_positives / predicted_positives
        return precision_item

    def slice(x, index):
        return x[index, :, :, :]

    f1_idx = K.constant(value=0.0, shape=())
    bs, w, h, chs = K.int_shape(y_true)
    for idx in range(FLAGS.batch_size):
        # tensor_slice with y_true and y_pred
        y_true_slice = Lambda(
            slice, output_shape=(w, h, chs), arguments={"index": idx})(y_true)
        y_pred_slice = Lambda(
            slice, output_shape=(w, h, chs), arguments={"index": idx})(y_pred)
        precision_item = precision(y_true_slice, y_pred_slice)
        recall_item = recall(y_true_slice, y_pred_slice)
        f1_idx += 2 * ((precision_item * recall_item) /
                       (precision_item + recall_item + K.epsilon()))
    return f1_idx / FLAGS.batch_size


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        y_true = K.flatten(K.argmax(y_true))
        y_pred = K.flatten(K.argmax(y_pred))
        true_positives = K.sum(y_true * y_pred)
        true_positives = K.cast(true_positives, "float32")
        possible_positives = K.sum(y_true)
        possible_positives = K.cast(possible_positives, "float32")
        recall = true_positives / possible_positives
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        y_true = K.flatten(K.argmax(y_true))
        y_pred = K.flatten(K.argmax(y_pred))
        true_positives = K.sum(y_true * y_pred)
        true_positives = K.cast(true_positives, "float32")
        predicted_positives = K.clip(
            K.cast(K.sum(y_pred), "float32"), K.epsilon(), None)
        # precision = K.cast(true_positives / predicted_positives, dtype='float32')
        precision = true_positives / predicted_positives
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def std_norm_along_chs(x):
    """Data normalization along the channle axis
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        xn = tensor4d, same shape as x, normalized version of x
    """
    avg = K.mean(x, axis=-1, keepdims=True)
    std = K.maximum(1e-4, K.std(x, axis=-1, keepdims=True))
    return (x - avg) / std


def atrous_spatial_pyramid_pooling(net,
                                   name,
                                   depth=128,
                                   rate_dict=[4, 8, 12, 16]):
    at_pool_dict = []
    for index, n_rate in enumerate(rate_dict):
        at_conv1 = Conv2D(
            depth,
            (3, 3),
            dilation_rate=(n_rate, n_rate),
            activation="linear",
            padding="same",
            kernel_regularizer=W_regularizer,
            name=name + "at1_%d" % index,
        )(net)
        at__bn1 = BatchNormalization(name=name + "%d_bn1" % index)(at_conv1)
        at__xn1 = Activation("relu", name=name + "%d_re1" % index)(at__bn1)
        at_conv2 = Conv2D(
            int(depth / 4),
            (1, 1),
            activation="linear",
            padding="same",
            kernel_regularizer=W_regularizer,
            name=name + "at2_%d" % index,
        )(at__xn1)
        at__bn2 = BatchNormalization(name=name + "%d_bn2" % index)(at_conv2)
        at__xn2 = Activation("relu", name=name + "%d_re2" % index)(at__bn2)
        at_pool_dict.append(at__xn2)
    xn = Concatenate(axis=3, name=name + "_concat")(at_pool_dict)
    return xn


def maxpooling_along_chs_opt(x):
    """Max Pooling along the feature dimension
    """
    return K.max(x, axis=3, keepdims=True)


def maxpooling_along_chs_shape(input_shape):
    n_samples, n_rows, n_cols, n_chs = input_shape
    output_shape = (n_samples, n_rows, n_cols, 1)
    return tuple(output_shape)


def avgpooling_along_chs_opt(x):
    """Average Pooling along the feature dimension
    """
    return K.mean(x, axis=3, keepdims=True)


def avgpooling_along_chs_shape(input_shape):
    n_samples, n_rows, n_cols, n_chs = input_shape
    output_shape = (n_samples, n_rows, n_cols, 1)
    return tuple(output_shape)


def sumpooling_along_chs_opt(x):
    """Average Pooling along the feature dimension
    """
    return K.sum(x, axis=1)


def sumpooling_along_chs_shape(input_shape):
    n_samples, n_rows, n_chs = input_shape
    output_shape = (n_samples, n_chs)
    return tuple(output_shape)


def channel_attention(x, name, inner_units_ratio=0.5):
    feature_map_shape = K.int_shape(x)
    channel_avg_weights = AveragePooling2D(
        (feature_map_shape[1], feature_map_shape[2]), (1, 1),
        name=name + "_ch_avg")(x)
    channel_max_weights = MaxPooling2D(
        (feature_map_shape[1], feature_map_shape[2]), (1, 1),
        name=name + "_ch_mxp")(x)
    channel_avg_reshape = Reshape((1,
                                   feature_map_shape[3]))(channel_avg_weights)
    channel_max_reshape = Reshape((1,
                                   feature_map_shape[3]))(channel_max_weights)
    channel_w_reshape = Concatenate(
        axis=1,
        name=name + "_ch_concat")([channel_avg_reshape, channel_max_reshape])
    fc1 = Dense(
        int(feature_map_shape[3] * inner_units_ratio),
        activation="relu",
        name=name + "_ch_fc1",
    )(channel_w_reshape)
    fc1_dp = Dropout(FLAGS.dp_rate)(fc1)
    fc2 = Dense(feature_map_shape[3], name=name + "_ch_fc2")(fc1_dp)
    fc2_dp = Dropout(FLAGS.dp_rate)(fc2)
    channel_attention = Lambda(sumpooling_along_chs_opt,
                               sumpooling_along_chs_shape)(fc2_dp)
    channel_attention = Activation("sigmoid")(channel_attention)
    channel_attention = Reshape((1, 1,
                                 feature_map_shape[3]))(channel_attention)
    feature_map_with_channel_attention = Multiply()([x, channel_attention])
    return feature_map_with_channel_attention


def spatial_attention(x, name):
    feature_map_shape = K.int_shape(x)
    channel_wise_avg_pooling = Lambda(avgpooling_along_chs_opt,
                                      avgpooling_along_chs_shape)(x)
    channel_wise_max_pooling = Lambda(maxpooling_along_chs_opt,
                                      maxpooling_along_chs_shape)(x)
    channel_wise_avg_pooling = Reshape(
        (feature_map_shape[1], feature_map_shape[2],
         1))(channel_wise_avg_pooling)
    channel_wise_max_pooling = Reshape(
        (feature_map_shape[1], feature_map_shape[2],
         1))(channel_wise_max_pooling)
    feature_map_concat = Concatenate(axis=3)(
        [channel_wise_avg_pooling, channel_wise_max_pooling])
    feature_map_conv = Conv2D(
        1,
        (7, 7),
        padding="same",
        activation="sigmoid",
        kernel_regularizer=W_regularizer,
        name=name + "_sp_conv",
    )(feature_map_concat)
    feature_map_with_attention = Multiply()([x, feature_map_conv])
    con_map = Add()([x, feature_map_with_attention])
    return con_map


def Conv2D_1x1_3x3(x, name, nb_inc, ksize=(3, 3)):
    uc = Conv2D(
        nb_inc,
        ksize,
        padding="same",
        kernel_regularizer=W_regularizer,
        activation="linear",
        name=name + "_uc",
    )(x)
    uc_norm = BatchNormalization(name=name + "_bn")(uc)
    xn = Activation("relu", name=name + "_re")(uc_norm)
    return xn


class SelfCorrelationPercPooling(Layer):
    """Custom Self-Correlation Percentile Pooling Layer
    Arugment:
        nb_pools = int, number of percentile poolings
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x_pool = tensor4d, (n_samples, n_rows, n_cols, nb_pools)
    """

    def __init__(self, nb_pools=115, **kwargs):
        self.nb_pools = nb_pools
        super(SelfCorrelationPercPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, x, mask=None):
        # parse input feature shape
        bsize, nb_rows, nb_cols, nb_feats = K.int_shape(x)
        nb_maps = nb_rows * nb_cols
        # normalization on channel-wise
        x_std = std_norm_along_chs(x)
        # self correlation
        x_3d = K.reshape(x_std, tf.stack([-1, nb_maps, nb_feats]))
        x_corr_3d = (tf.matmul(
            x_3d, x_3d, transpose_a=False, transpose_b=True) / nb_feats)
        x_corr = K.reshape(x_corr_3d, tf.stack([-1, nb_rows, nb_cols,
                                                nb_maps]))
        # argsort response maps along the translaton dimension
        if self.nb_pools is not None:
            ranks = K.cast(
                K.round(tf.lin_space(1.0, nb_maps - 1, self.nb_pools)),
                "int32")
        else:
            ranks = tf.range(1, nb_maps, dtype="int32")
        x_sort, _ = tf.nn.top_k(x_corr, k=nb_maps, sorted=True)
        # pool out x features at interested ranks
        x_f1st_sort = K.permute_dimensions(x_sort, (3, 0, 1, 2))
        x_f1st_pool = tf.gather(x_f1st_sort, ranks)
        x_pool = K.permute_dimensions(x_f1st_pool, (1, 2, 3, 0))
        return x_pool

    def compute_output_shape(self, input_shape):
        bsize, nb_rows, nb_cols, nb_feats = input_shape
        nb_pools = (self.nb_pools if (self.nb_pools is not None) else
                    (nb_rows * nb_cols - 1))
        return tuple([bsize, nb_rows, nb_cols, nb_pools])


class BilinearUpSampling2D(Layer):
    """Custom 2x bilinear upsampling layer
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x2 = tensor4d, (n_samples, 2*n_rows, 2*n_cols, n_feats)
    """

    def call(self, x, mask=None):
        bsize, nb_rows, nb_cols, nb_filts = K.int_shape(x)
        new_size = tf.constant([nb_rows * 2, nb_cols * 2], dtype=tf.int32)
        return tf.image.resize_bilinear(x, new_size, align_corners=True)

    def compute_output_shape(self, input_shape):
        bsize, nb_rows, nb_cols, nb_filts = input_shape
        return tuple([bsize, nb_rows * 2, nb_cols * 2, nb_filts])


class Preprocess(Layer):
    """Basic preprocess layer for BusterNet

    More precisely, it does the following two things
    1) normalize input image size to (256,256) to speed up processing
    2) substract channel-wise means if necessary
    """

    def call(self, x, mask=None):
        _, h, w, _ = K.int_shape(x)
        new_size = [IMAGE_SIZE, IMAGE_SIZE]
        if h != IMAGE_SIZE or w != IMAGE_SIZE:
            x = tf.image.resize_bilinear(x, new_size, align_corners=True)
        # substract channel means if necessary
        print("VGG_Preprocess_input...")
        xout = preprocess_input(x, mode="tf")
        return xout

    def compute_output_shape(self, input_shape):
        return (input_shape[0], IMAGE_SIZE, IMAGE_SIZE, 3)


class ResizeBack(Layer):
    def call(self, x):
        new_size = [512, 512]
        return tf.image.resize_bilinear(x, new_size, align_corners=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 512, 512, input_shape[-1])


class ResizeBack_for_unsize(Layer):
    def call(self, x):
        t, r = x
        new_size = [tf.shape(r)[1], tf.shape(r)[2]]
        return tf.image.resize_images(t, new_size, align_corners=True)

    def compute_output_shape(self, input_shapes):
        tshape, rshape = input_shapes
        return (tshape[0],) + rshape[1:3] + (tshape[-1],)


def simi_cfmd(img_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="simiDet"):
    # input
    img_input = Input(shape=img_shape, name=name + "_in")
    # vgg16_layers_f4
    bname = name + "_cnn"
    # Block 1
    x1 = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=W_regularizer,
        name=bname + "_b1c1",
    )(img_input)
    x1 = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=W_regularizer,
        name=bname + "_b1c2",
    )(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name=bname + "_b1p")(x1)
    # Block 2
    x2 = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=W_regularizer,
        name=bname + "_b2c1",
    )(x1)
    x2 = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=W_regularizer,
        name=bname + "_b2c2",
    )(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name=bname + "_b2p")(x2)
    # Block 3
    x3 = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=W_regularizer,
        name=bname + "_b3c1",
    )(x2)
    x3 = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=W_regularizer,
        name=bname + "_b3c2",
    )(x3)
    x3 = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=W_regularizer,
        name=bname + "_b3c3",
    )(x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name=bname + "_b3p")(x3)
    # Block 4
    x4 = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=W_regularizer,
        name=bname + "_b4c1",
    )(x3)
    x4 = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        dilation_rate=(2, 2),
        kernel_regularizer=W_regularizer,
        name=bname + "_b4c2",
    )(x4)
    x4 = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        dilation_rate=(4, 4),
        kernel_regularizer=W_regularizer,
        name=bname + "_b4c3",
    )(x4)
    # SAM
    x4_att = channel_attention(x4, "x4_ch_att")
    x3_att = channel_attention(x3, "x3_ch_att")
    # self_correlation
    fnx4 = SelfCorrelationPercPooling(name=bname + "_corr_x4")(x4_att)
    fnx3 = SelfCorrelationPercPooling(name=bname + "_corr_x3")(x3_att)
    fn_x = Concatenate(name=bname + "_concat_fn")([fnx4, fnx3])
    # BatchNormalization
    f_xn = BatchNormalization(name=bname + "_bn")(fn_x)
    # Reduce dims
    f_xn_d = Conv2D_1x1_3x3(f_xn, bname + "f_xn_d", 115, (1, 1))  # 64
    # ASPP
    xn_as = atrous_spatial_pyramid_pooling(
        f_xn_d, depth=230, name=bname + "_aspp")  # 128
    # Deconv
    bname = name + "_dconv"
    # upsampling 64x64
    f32 = Conv2D_1x1_3x3(xn_as, bname + "_f64", 64)
    f32 = spatial_attention(f32, "f64_sp_att")
    f64 = BilinearUpSampling2D(name=bname + "_up64")(f32)
    # upsampling 128x128
    f128 = Conv2D_1x1_3x3(f64, bname + "_f128", 32)
    f128 = spatial_attention(f128, "f128_sp_att")
    f128 = BilinearUpSampling2D(name=bname + "_up128")(f128)
    # upsampling 256x256
    f256 = Conv2D_1x1_3x3(f128, bname + "_f256", 16)
    f256 = spatial_attention(f256, "f256_sp_att")
    f256 = BilinearUpSampling2D(name=bname + "_up256")(f256)
    # summary
    f256_final = Conv2D(
        2,
        (3, 3),
        activation="softmax",
        name=bname + "_f256_final",
        kernel_regularizer=W_regularizer,
        padding="same",
    )(f256)
    model = Model(inputs=img_input, outputs=f256_final, name=name)
    return model


def main(argv=None):
    training_records, validation_records, test_records = creat_dataset.read_dataset(
        "pickle_files", "simi")
    # all_records = training_records + validation_records
    validation_records = validation_records[:int(len(validation_records) / FLAGS.batch_size) *
                                FLAGS.batch_size]
    SAMPLES_TRAIN = len(training_records)
    SAMPLES_VALID = len(validation_records)
    NUM_TRAIN_STEPS = int(np.ceil(SAMPLES_TRAIN / FLAGS.batch_size))
    NUM_VALID_STEPS = int(np.ceil(SAMPLES_VALID / FLAGS.batch_size))
    print("NO. %d of traininig samples" % SAMPLES_TRAIN)
    print("NO. %d of validation samples" % SAMPLES_VALID)
    # simi_fn
    simi = simi_cfmd()
    SimiDet = Model(
        inputs=simi.inputs, outputs=simi.layers[-1].output, name="simiFeatex")

    img_raw = Input(shape=(None, None, 3), name="image_in")
    img_in = Preprocess(name="preprocess")(img_raw)
    simi_feat = SimiDet(img_in)
    train_model = Model(inputs=img_raw, outputs=simi_feat, name="simiDet")

    # opt = keras.optimizers.SGD(
    #     lr=FLAGS.learning_rate, momentum=0.9, nesterov=True)
    opt = keras.optimizers.Adam(lr=FLAGS.learning_rate)
    train_model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["acc", f1, f1_avg, iou],
    )
    # keras 2.1.0
    # reduce_lr_plateau = ReduceLROnPlateau(
    #     monitor="val_loss",
    #     factor=FLAGS.decay_rate,
    #     patience=1,
    #     verbose=1,
    #     mode="min",
    #     epsilon=1e-3,
    #     cooldown=0,
    #     min_lr=1e-5,
    # )
    # keras 2.2.0
    reduce_lr_plateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=FLAGS.decay_rate,
        patience=1,
        verbose=1,
        mode="min",
        min_delta=1e-3,
        cooldown=0,
        min_lr=1e-5,
    )
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(FLAGS.weights_path,
                              "weights_{epoch:02d}-{val_acc:.3f}.hdf5"),
        save_weights_only=True,
        verbose=1,
        save_best_only=False,
        mode="min",
    )
    # earlystopping = EarlyStopping(patience=2)
    path = os.path.join(FLAGS.weights_path, FLAGS.model_weights)
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
    train_model.fit_generator(
        generator=read_images(training_records, SAMPLES_TRAIN, FLAGS.batch_size),
        steps_per_epoch=NUM_TRAIN_STEPS,
        epochs=FLAGS.NUM_EPOCHS_TRAIN,
        validation_data=read_images(
            validation_records, SAMPLES_VALID, FLAGS.batch_size, train=False),
        validation_steps=NUM_VALID_STEPS,
        callbacks=[checkpointer, reduce_lr_plateau],
    )
    # save Model
    train_model.save_weights(
        os.path.join(FLAGS.weights_path, "train_all_weights.h5"))


def test_model(model_h5, is_resizeback=False):
    # simi_fn
    simi = simi_cfmd()
    SimiDet = Model(
        inputs=simi.inputs, outputs=simi.layers[-1].output, name="simiFeatex")

    img_in = Input(shape=(None, None, 3), name="image_in")
    img_raw = Preprocess(name="preprocess")(img_in)
    simi_feat = SimiDet(img_raw)
    if is_resizeback:
        simi_feat = ResizeBack(name="resizeback")(simi_feat)
    train_model = Model(inputs=img_in, outputs=simi_feat, name="simiDet")
    if model_h5 is not None:
        try:
            train_model.load_weights(model_h5)
            print("INFO: successfully load pretrained weights from {}".format(
                model_h5))
        except Exception as e:
            print(
                "INFO: fail to load pretrained weights from {} for reason: {}".
                format(model_h5, e))
    return train_model


if __name__ == "__main__":
    tf.app.run()
