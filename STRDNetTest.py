import cv2 as cv
import os
import warnings
import numpy as np
from scipy import misc
from PIL import Image
from STRDNet import creat_test_model as model_target_source

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings("ignore")

# parms !!!
CUT_ENLARGE = 5 # 5 15 25
KERNEL_SIZE_NOISE_DOWN = 3 # 5
DIST_RATE = 0.55
DIST_RATE_COVERAGE = 0.68


def creat_arr_ts(index, test_model_st, mask, image, dist_rate):
    output_mask = np.zeros((256, 256, 3), dtype=np.uint8)

    # mask = np.array(mask).astype(bool)
    # mask = morphology.remove_small_objects(mask, min_size=200, connectivity=1)
    # mask = (np.array(mask) * 255).astype(float) 

    kernel = np.ones((KERNEL_SIZE_NOISE_DOWN, KERNEL_SIZE_NOISE_DOWN), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv.distanceTransform(np.uint8(mask), cv.DIST_L2, 5)
    _, mask_res = cv.threshold(dist_transform, dist_rate * dist_transform.max(), 1.0, 0)

    # _, mask_res = cv.threshold(mask, 0.5, 1.0, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    mask_res = cv.erode(mask_res, element, iterations = 1)
    mask_res = cv.dilate(mask_res, element, iterations = 1)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(np.uint8(mask_res), 4, cv.CV_32S)

    if num_labels < 2:
        return None
    
    if len(stats) > 2:
        start_index = 1
    else:
        start_index = 0
    patches = []
    for i in range(start_index, len(stats)):
        stat = stats[i]
        patch = image[max(stat[1]-CUT_ENLARGE,0):min(stat[1]+stat[3]+CUT_ENLARGE,256),max(stat[0]-CUT_ENLARGE,0):min(stat[0]+stat[2]+CUT_ENLARGE,256),...]
        if patch.shape != (128, 128, 3):
            patch = misc.imresize(patch, [128, 128], interp="bilinear")
        # misc.imsave("patch_%d.png" % i, patch)
        patches.append(np.expand_dims(patch, 0))
    if len(patches) < 2:
        return None

    patches = np.concatenate(patches, 0)
    predict_probs = test_model_st.predict(patches, verbose=0)

    poss_source_min = 0.0
    min_prob_index = 0
    for i in range(predict_probs.shape[0]):
        if predict_probs[i][0] > poss_source_min:
            poss_source_min = predict_probs[i][0]
            min_prob_index = i

    tampered_stat = stats[start_index+min_prob_index]
    res_tampered = np.zeros((256, 256), dtype=np.uint8)
    for x in range(max(0, tampered_stat[1]-10),min(tampered_stat[1]+tampered_stat[3]+10, 256)):
        for y in range(max(0, tampered_stat[0]-10),min(tampered_stat[0]+tampered_stat[2]+10, 256)):
            if mask[x][y]==255.0:
                res_tampered[x,y] = 255.0
    res_green = res_tampered == 255.0
    output_mask[res_green, 1] = 255.0
    res_red = np.array(mask-res_tampered,np.bool)
    output_mask[res_red, 0] = 255.0
    for x in range(max(0, tampered_stat[1]-CUT_ENLARGE-10),min(tampered_stat[1]+tampered_stat[3]+CUT_ENLARGE+10, 256)):
        for y in range(max(0, tampered_stat[0]-CUT_ENLARGE-10),min(tampered_stat[0]+tampered_stat[2]+CUT_ENLARGE+10, 256)):
            if output_mask[:,:,0][x][y]==255.0:
                output_mask[:,:,1][x,y] = 255.0
                output_mask[:,:,0][x,y] = 0.0
    return output_mask


if __name__ == "__main__":
    test_model_st = model_target_source("model/model_h5_STRDNet.hdf5")
    predict_map_CMSDNet = misc.imread("predict_map_CMSDNet.png")
    image_map = misc.imread("test.png")
    pred_mask = creat_arr_ts(0, test_model_st, predict_map_CMSDNet, image_map, DIST_RATE)
    if pred_mask is None:
        print("failed!")
    else:
        misc.imsave("predict_map_STRDNet.png", pred_mask)
