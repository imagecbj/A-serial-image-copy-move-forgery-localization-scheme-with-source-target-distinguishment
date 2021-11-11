from __future__ import print_function
import os, time
import numpy as np
from scipy import misc
from skimage import morphology
from simi_keras_perason_fm_115_run1 import test_model
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# weights_03-0.970.hdf5 -- F1: 0.6850
model_weights = "weights_03-0.970.hdf5"


def precision_recall_fscore_binary(pred_logits, annotation, min_size):
    print(min_size)
    # annotation = np.squeeze(annotation, axis=3)
    # annotation = annotation[:, :, :, 0] + annotation[:, :, :, 1]
    prf_list = []
    i = 0
    for rr, hh in zip(annotation, pred_logits):
        # misc.imsave("data_zoo/COVERAGE/images_91/{}.tif".format(i), ii)
        # misc.imsave("data_zoo/COVERAGE/preds_91/{}.tif".format(i), hh)
        hh = np.array(hh).astype(bool)
        hh = morphology.remove_small_objects(
            hh, min_size=min_size, connectivity=1
        )
        hh = np.array(hh).astype(int)
        ref = rr.flatten() == 255.0
        hyf = hh.flatten() == 1.0
        # ref = rr.flatten() > 127.5
        # hyf = hh.flatten() == 1.0
        precision, recall, fscore, _ = precision_recall_fscore_support(
            ref, hyf, pos_label=1, average='binary')
        prf_list.append([precision, recall, fscore])
        if fscore > 0.5:
            i += 1
    prf = np.row_stack(prf_list)
    for name, mu in zip(['Precision', 'Recll', 'F1'], prf.mean(axis=0)):
        print("INFO: {:>9s} = {:.4f}".format(name, mu))
    print("F_measure > 0.5: {} / {}".format(i, annotation.shape[0]))
    return prf.mean(axis=0)[-1]


def precision_recall_fscore_binary1(pred_logits, annotation):
    # annotation = annotation[:, :, :, 0] + annotation[:, :, :, 1]
    prf_list = []
    for rr, hh in zip(annotation, pred_logits):
        ref = rr.flatten() == 255.0
        hyf = hh.flatten() == 1.0
        precision, recall, fscore, _ = precision_recall_fscore_support(
            ref, hyf, pos_label=1, average='binary')
        prf_list.append([precision, recall, fscore])
    prf = np.row_stack(prf_list)
    for name, mu in zip(['Precision', 'Recll', 'F1'], prf.mean(axis=0)):
        print("INFO: {:>9s} = {:.4f}".format(name, mu))

def fs_(pred, gt):
    ref = gt.flatten() == 255.0
    hyf = pred.flatten() == 1.0
    _, _, fscore, _ = precision_recall_fscore_support(
    ref, hyf, pos_label=1, average='binary')
    return fscore
test_model_d = test_model('logs/hdf5/weights_perason_fm_115/run1/model/' + model_weights, True)
imgslist = os.listdir('data_zoo/dataset/images/test')
fss = []
for index, img in zip(range(0, len(imgslist)), imgslist):
    imgname = img.split('/')[-1]
    image = misc.imread('data_zoo/dataset/images/test/' + img)
    pred = test_model_d.predict(np.expand_dims(image, axis=0), verbose=0, batch_size=1)
    pred = np.argmax(pred, axis=-1)
    gt = misc.imread('data_zoo/dataset/labels_masks/test/' + img)
    gt = gt[:, :, 0] + gt[:, :, 1]
    fs = fs_(pred, gt)
    fss.append(fs)
    del image, pred, gt
    if index % 1000 == 0:
        print(index)
print('{:.4f}'.format(np.mean(fss)))
# X_CMF = HDF5Matrix('data_zoo/test_10100.hd5', 'X')
# Y_CMF = HDF5Matrix('data_zoo/test_10100.hd5', 'Y2')
# Y_CMF = np.array(Y_CMF)
# print(Y_CMF.shape)
# Z_CMF = test_model_d.predict(X_CMF, verbose=1, batch_size=32)
# Z_CMF = np.argmax(Z_CMF, axis=-1)
# precision_recall_fscore_binary1(Z_CMF, Y_CMF)
# min_sizes = [10, 20, 30, 40]
# for min_size in min_sizes:
# precision_recall_fscore_binary1(Z_CMF, Y_CMF)
# for min_size in range(10, 100, 4):  # CoMoFoD: min_size=426, COBERAGE: min_size=10, casia: min_size=0
#     pred_f1 = precision_recall_fscore_binary(X_CMF, Z_CMF, Y_CMF, min_size=min_size)
