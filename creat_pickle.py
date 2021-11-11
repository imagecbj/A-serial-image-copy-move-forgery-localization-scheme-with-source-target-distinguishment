import os
import random
from six.moves import cPickle as pickle
from scipy import misc
import tensorflow as tf
import numpy as np
import glob
import h5py

DATA_FLODER = "data_zoo/dataset"  # 包含数据集的文件夹名


def read_dataset(data_dir, pickle_filename):
    '''
    data_dir:文件路径;
    文件夹：总文件夹DATA_FLODER下包含annotations和images两个文件，annotations和images下分别包含training和valitation两个文件夹;
    文件名:images和annotations的文件名需对应相同 .
    '''
    pickle_filepath = os.path.join(data_dir, pickle_filename)  # pickle文件路径地址
    # print(pickle_filepath)
    if not os.path.exists(pickle_filepath):  # 判断pickle文件是否存在,不存在写入文件
        result = creat_image_lists(DATA_FLODER)  # 把所有文件写入列表result
        print("Pickling ...")
        with open(pickle_filepath, 'wb') as f:  # 写入pickle文件
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Found pickle file...")
    with open(pickle_filepath, 'rb') as f:  # 读取pickle文件
        result = pickle.load(f)
        records = result["CASIA_2_CP"]
        del result  # 释放空间
    return records


def creat_image_lists(image_dir):
    if not tf.gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    else:
        print("Data_images_folder_file existed...")

    directories = ['CASIA_2_CP']  # 训练集和验证集
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, directory, "image",
                                 "*.tif")  # 所有的文件路径
        # print(file_glob)
        file_list.extend(glob.glob(
            file_glob))  # 把文件夹下匹配的所有文件路径加入到file_list列表中，文件路径名是file_glob

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                filename = filename.split('\\')[-1]
                # print(filename)
                annotation_file = os.path.join(
                    image_dir, directory,
                    'mask/{}.png'.format(filename))  # 原图对应的annotation文件路径
                # print(annotation_file)
                if os.path.exists(annotation_file):
                    record = {
                        "image": f,  # 原图文件路径
                        "annotation": annotation_file,  # 标签文件路径
                        "filename": filename  # 文件名
                    }
                    image_list[directory].append(record)  # 训练或验证集列表加入字典元素
                else:
                    print("Annotation file not found for %s - Skipping" %
                          filename)
        random.shuffle(image_list[directory])  # 打乱数据集
        len_of_images = len(image_list[directory])
        print("No. of %s files: %d" % (directory, len_of_images))

    return image_list


class BatchDatset:
    # 数据集类中固定的一些属性
    files = []
    images = []
    annotations = []

    def __init__(self, records_list):
        self.files = records_list
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([
            self.transform_image(filename['image']) for filename in self.files
        ])
        self.__channels = False
        self.annotations = np.array([
            np.expand_dims(self.transform_image(filename['annotation']),
                           axis=3) for filename in self.files
        ])
        print(self.images.shape)
        print(self.annotations.shape)

    def transform_image(self, filename):
        image = misc.imread(filename)
        if (image.shape[0], image.shape[1]) != (1024, 1024):
            image_resize = misc.imresize(image, [1024, 1024],
                                         interp='bilinear')
        else:
            image_resize = image
        if self.__channels:
            out_image = image_resize
        else:
            image_resize = image_resize[:, :, 0] + image_resize[:, :, 1]
            image_resize[image_resize > 0] = 255
            out_image = image_resize / 255
        return np.array(np.float32(out_image))

    def read_all(self):
        return self.images[:], self.annotations[:]


if __name__ == "__main__":
    pickle_filename = "CASIA_FOR_HIGH_RESOLUTION.pickle"
    records = read_dataset('pickle_files', pickle_filename)
    print(records[0])
    dataset = BatchDatset(records)
    images, annotations = dataset.read_all()
    with h5py.File('data_zoo/CASIA_FOR_HIGH_RESOLUTION.hd5') as f:
        f['X'] = images
        f['Y'] = annotations
        f.close()
