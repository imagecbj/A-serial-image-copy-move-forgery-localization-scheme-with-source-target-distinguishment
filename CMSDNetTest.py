import os, time
import numpy as np
from scipy import misc
from skimage import morphology
from CMSDNet import test_model
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

test_model_d = test_model('model/model_h5_CMSDNet.hdf5')
image = misc.imread("test.png")
image = np.expand_dims(np.array(image), 0)
predict_map = test_model_d.predict(image, verbose=0, batch_size=1)
predict_map = np.argmax(predict_map, -1)
predict_map = np.squeeze(predict_map, 0)

# predict_map = np.array(predict_map).astype(bool)
# predict_map = morphology.remove_small_objects(predict_map, min_size=1, connectivity=1)
# predict_map = np.array(predict_map).astype(int)

misc.imsave("predict_map_CMSDNet.png", predict_map)