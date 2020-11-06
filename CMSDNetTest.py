import os, time
import numpy as np
from scipy import misc
from skimage import morphology
from CMSDNet import test_model
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

test_model_d = test_model('model/model_h5_CMSDNet.hdf5')
# input image
image = misc.imread("test.png")
image = np.expand_dims(np.array(image), 0)
predict_map = test_model_d.predict(image, verbose=1, batch_size=1)
predict_map = np.argmax(predict_map, -1)
# save
misc.imsave("predict_map_CMSDNet.png", np.squeeze(predict_map, 0))
