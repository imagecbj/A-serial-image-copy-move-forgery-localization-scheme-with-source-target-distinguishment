import os
import random
from six.moves import cPickle as pickle
import tensorflow as tf
import glob

DATA_FLODER = "data_zoo/dataset"  # 包含数据集的文件夹名


def read_dataset(data_dir, data_branch):
    '''
    data_dir:文件路径;
    文件夹：总文件夹DATA_FLODER下包含annotations和images两个文件，annotations和images下分别包含training和valitation两个文件夹;
    文件名:images和annotations的文件名需对应相同 .
    '''
    if data_branch == "simi":
        pickle_filename = "train_valid_test_simi.pickle"
    elif data_branch == "mani":
        pickle_filename = "train_valid_test_mani.pickle"
    elif data_branch == "target_source":
        pickle_filename = "target_source_det.pickle"
    else:
        pickle_filename = "train_valid_test_all.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)  # pickle文件路径地址
    # print(pickle_filepath)
    if not os.path.exists(pickle_filepath):  # 判断pickle文件是否存在,不存在写入文件
        result = creat_image_lists(os.path.join(data_dir, DATA_FLODER))  # 把所有文件写入列表result
        print("Pickling ...")
        with open(pickle_filepath, 'wb') as f:  # 写入pickle文件
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Found pickle file...")
    with open(pickle_filepath, 'rb') as f:  # 读取pickle文件
        result = pickle.load(f)
        training_records = result["training"]
        validation_records = result["validation"]
        test_records = result["test"]
        del result  # 释放空间
    return training_records, validation_records, test_records


def creat_image_lists(image_dir):
    if not tf.gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    else:
        print("Data_images_folder_file existed...")

    directories = ['training', 'validation', 'test']  # 训练集和验证集
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, "*.tif")  # 所有的文件路径
        # print(file_glob)
        file_list.extend(glob.glob(file_glob))  # 把文件夹下匹配的所有文件路径加入到file_list列表中，文件路径名是file_glob
        
        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                filename = filename.split('\\')[-1]
                # print(filename)
                annotation_file = os.path.join(image_dir, "labels_three_classes", directory, filename + '.tif')  # 原图对应的annotation文件路径
                # print(annotation_file)
                if os.path.exists(annotation_file):
                    record = {
                        "image": f,  # 原图文件路径
                        "annotation": annotation_file,  # 标签文件路径
                        "filename": filename  # 文件名
                    }
                    image_list[directory].append(record)  # 训练或验证集列表加入字典元素
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        random.shuffle(image_list[directory])  # 打乱数据集
        len_of_images = len(image_list[directory])
        print("No. of %s files: %d" % (directory, len_of_images))

    return image_list


if __name__ == "__main__":
    a, b, c = read_dataset('pickle_files', 'target_source')
    print(len(a), len(b), len(c))
    print(a[:2], b[:2], c[:2])