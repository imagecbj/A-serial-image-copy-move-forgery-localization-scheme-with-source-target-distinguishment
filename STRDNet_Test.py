import h5py
import cv2 as cv
import os
import warnings
import numpy as np
import time
from scipy import misc
from PIL import Image
from target_source_Bayar_for_100k import creat_test_model as model_target_source, FLAGS
from copy import copy
from skimage import morphology
from simi_keras_perason_fm_115 import test_model
from sklearn.metrics import precision_recall_fscore_support

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

model_path = (
    "logs/hdf5/weights_target_source/target_source_Bayar/model/" + FLAGS.model_name
)
# some hyper-parms in preprocessing
CUT_ENLARGE = 15  # init=10 CoMoFoD=15
KERNEL_SIZE_NOISE_DOWN = 3
DIST_RATE = 0.55
DIST_RATE_COVERAGE = 0.68

def check_one_sample(z, y):
    """Check dataset's Discernibility for one sample
    Input:
        z = np.array, dataset predicted mask
        y = np.array, GT mask
    Output:
        src_label = the dominant class on the src copy, if 0 then correct
        dst_label = the dominant class on the dst copy, if 1 then correct
    """

    def hist_count(arr):
        nb_src = np.sum(arr == 0)
        nb_dst = np.sum(arr == 1)
        nb_bkg = np.sum(arr == 2)
        return [nb_src, nb_dst, nb_bkg]

    def get_label(hist):
        if np.sum(hist[:2]) == 0:
            return 2
        else:
            return np.argmax(hist[:2])

    # 1. determine pixel membership from the probability map
    hyp = z.argmax(axis=-1)
    # 2. get the gt src/dst masks
    ref_src = y[..., 0] > 0.5
    ref_dst = y[..., 1] > 0.5
    # 3. count the membership histogram on src/dst masks respectively
    src_hist = hist_count(hyp[ref_src])
    src_label = get_label(src_hist)
    dst_hist = hist_count(hyp[ref_dst])
    dst_label = get_label(dst_hist)
    return src_label, dst_label

def creat_arr_ts(index, test_model_st, mask, image, dist_rate):
    # 最终的区分图RGB
    output_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    output_mask[:,:,2] = 255

    # predict background
    res = mask == 255.0
    # 去除mask区域，其余区域是背景
    output_mask[res, 2] = 0

    # 应用一些形态学处理方式
    # 去除小的孤立的噪点
    # mask = np.array(mask).astype(bool)
    # mask = morphology.remove_small_objects(
    #     mask, min_size=200, connectivity=1
    # )
    # mask = (np.array(mask) * 255).astype(float) 

    kernel = np.ones((KERNEL_SIZE_NOISE_DOWN, KERNEL_SIZE_NOISE_DOWN), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv.distanceTransform(np.uint8(mask), cv.DIST_L2, 5)
    _, mask_res = cv.threshold(
        dist_transform, dist_rate * dist_transform.max(), 1.0, 0
    )

    # _, mask_res = cv.threshold(mask, 0.5, 1.0, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    mask_res = cv.erode(mask_res, element, iterations = 1)
    mask_res = cv.dilate(mask_res, element, iterations = 1)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(np.uint8(mask_res), 4, cv.CV_32S)

    if num_labels < 3:
        return None

    patches = []
    for i in range(1, len(stats)):
        stat = stats[i]
        patch = image[max(stat[1]-CUT_ENLARGE,0):min(stat[1]+stat[3]+CUT_ENLARGE,256),max(stat[0]-CUT_ENLARGE,0):min(stat[0]+stat[2]+CUT_ENLARGE,256),...]
        if patch.shape != (128, 128, 3):
            patch = misc.imresize(patch, [128, 128], interp="bilinear")
        # print(patch.shape)
        patches.append(np.expand_dims(patch, 0))
    if len(patches) < 2:
        return None

    # 预测每个patch是篡改的概率
    patches = np.concatenate(patches, 0)
    predict_probs = test_model_st.predict(patches, verbose=0)

    # 获取置信度最高的未篡改patch的下标
    poss_source_min = 0.0
    min_prob_index = 0
    for i in range(predict_probs.shape[0]):
        # for axis = 0
        if predict_probs[i][0] > poss_source_min:  # min(y) untampered-source region
            poss_source_min = predict_probs[i][0]
            min_prob_index = i
    # print(min_prob_index)

    tampered_stat = stats[1+min_prob_index]
    res_tampered = np.zeros((256, 256), dtype=np.uint8)
    def change(x,y):
        res_tampered[x,y] = 255.0
    [change(x,y) for x in range(max(0, tampered_stat[1]-10),min(tampered_stat[1]+tampered_stat[3]+10, 256)) for y in range(max(0, tampered_stat[0]-10),min(tampered_stat[0]+tampered_stat[2]+10, 256)) if mask[x][y]==255.0]
    res_green = res_tampered == 255.0
    output_mask[res_green, 1] = 255.0
    res_red = np.array(mask-res_tampered,np.bool)
    output_mask[res_red, 0] = 255.0
    # misc.imsave('figs/show%d.png'%index, output_mask)
    return output_mask

def precision_recall_fscore_binary(pred_logits, annotation):
    annotation = annotation[:, :, :, 0] + annotation[:, :, :, 1]
    prf_list = []
    for rr, hh in zip(annotation, pred_logits):
        hh = np.array(hh).astype(bool)
        hh = morphology.remove_small_objects(
            hh, min_size=200, connectivity=1
        )
        hh = np.array(hh).astype(int)
        ref = rr.flatten() == 255.0
        hyf = hh.flatten() == 1.0
        precision, recall, fscore, _ = precision_recall_fscore_support(
            ref, hyf, pos_label=1, average='binary')
        prf_list.append([precision, recall, fscore])
    prf = np.row_stack(prf_list)
    for name, mu in zip(['Precision', 'Recll', 'F1'], prf.mean(axis=0)):
        print("INFO: {:>9s} = {:.4f}".format(name, mu))

if __name__ == "__main__":
    test_model_st = model_target_source(model_path)
    #对数据集进行批量测试 --源/目标是否正确区分
    # CoMoFoD_attacks_test 包含binary gt图, CoMoFoD_attacks_test_st 包含 3classes mask图
    f_casia = h5py.File("data_zoo/CASIA_image_mask_pred_simi_ts.hd5")
    # pred, X, Y --CASIA CoMoFoD COVERAGE
    pred_casia, image_casia, label_casia = (f_casia["pred"], f_casia["X"], f_casia["Y"])
    # pred, X, Y1 --test10100
    # pred_casia, image_casia, label_casia = (f_casia["pred"], f_casia["X"], f_casia["Y1"])
    correct = 0
    for index, (p_st, im_st, y) in enumerate(zip(pred_casia, image_casia, label_casia)):
        image = im_st
        mask = p_st
        pred_mask = creat_arr_ts(index, test_model_st, mask, image, DIST_RATE)
        if pred_mask is not None:
            src_label, dst_label = check_one_sample(pred_mask, y)
            if src_label == 0 and dst_label == 1:
                correct += 1
    print(correct)
    '''
    # 攻击对比
    attack_txt = open('attack_st.txt', 'w')
    f_imgs = h5py.File("data_zoo/CoMoFoD_attacks_test_st.hd5")
    f_attacks = h5py.File("data_zoo/CoMoFoD_attacks_simipred.hd5")
    yy = f_imgs["mask"]
    attacks = [
        "F",
        "BC1",
        "BC2",
        "BC3",
        "CA1",
        "CA2",
        "CA3",
        "CR1",
        "CR2",
        "CR3",
        "IB1",
        "IB2",
        "IB3",
        "JC1",
        "JC2",
        "JC3",
        "JC4",
        "JC5",
        "JC6",
        "JC7",
        "JC8",
        "JC9",
        "NA1",
        "NA2",
        "NA3",
        # "mask",
        # "B_L",
        ]
    for attack in attacks:
        attack_images = f_imgs[attack]
        bin_gt = f_attacks[attack]
        correct = 0
        for index, (p_st, im_st, y) in enumerate(zip(bin_gt, attack_images, yy)):
            image = im_st
            mask = p_st
            pred_mask = creat_arr_ts(index, test_model_st, mask, image, DIST_RATE_CoMoFoD)
            if pred_mask is not None:
                src_label, dst_label = check_one_sample(pred_mask, y)
                if src_label == 0 and dst_label == 1:
                    correct += 1
        print(attack, correct)
        attack_txt.write('%s-%d \n' % (attack, correct))
    '''
