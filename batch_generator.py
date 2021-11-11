import numpy as np
import random
from scipy import misc
from keras.utils import to_categorical


def read_images(records, samples, batch_size, train=True):
    while True:
        if train:
            random.shuffle(records)
        for i in range(0, samples, batch_size):
            train_list = records[i:i+batch_size]
            channels = True
            images = np.array([transform_image(filename['image'], channels) for filename in train_list])
            channels = False
            annotations = np.array(
                [np.expand_dims(transform_image(filename['annotation'], channels), axis=3) for filename in train_list])
            annotations = to_categorical(annotations, num_classes=2)
            # print(images.shape)
            # print(annotations.shape)
            # Shuffle the data
            if train:
                perm = np.arange(images.shape[0])
                np.random.shuffle(perm)
                images = images[perm]
                annotations = annotations[perm]
            yield (images, annotations)


def read_validations(images, annotations, samples, batch_size):
    while True:
        for i in range(0, samples, batch_size):
            image = images[i:i+batch_size]
            annotation = annotations[i:i+batch_size]
            annotation = to_categorical(annotation, num_classes=2)
            yield (image, annotation)


def transform_image(filename, channels):
    image = misc.imread(filename)
    if channels:
        h, w, c = image.shape
        if h != 256 or w != 256:
            image_resize = misc.imresize(image, [256, 256], interp='bilinear')
        else:
            image_resize = image
        out_image = np.float32(image_resize)
    else:
        out_image = image / 255
    return np.array(np.float32(out_image))
