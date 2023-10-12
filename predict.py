import argparse
import glob
from time import *
import numpy as np
from PIL import Image
from keras.optimizers import adam_v2
import imageio
import os
import resunet_up
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import tensorflow.compat.v1 as tf
from utils.utils import prctile_norm, rm_outliers
from keras.utils import plot_model


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="test")
parser.add_argument("--folder_test", type=str, default="images")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--gpu_memory_fraction", type=float, default=0.7)
parser.add_argument("--model_name", type=str, default="resu")
parser.add_argument("--model_weights", type=str, default="weight/weights.best.h5")
parser.add_argument("--input_height", type=int, default=512)
parser.add_argument("--input_width", type=int, default=512)
parser.add_argument("--scale_factor", type=int, default=2)


args = parser.parse_args()
gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction
data_dir = args.data_dir
folder_test = args.folder_test
model_name = args.model_name
model_weights = args.model_weights
input_width = args.input_width
input_height = args.input_height
scale_factor = args.scale_factor

data_name = model_weights.split('/')[-2]
save_weights_name = model_name + '-SIM_' + data_name
output_name = 'output_' + save_weights_name + '-'
test_images_path = data_dir + '/' + folder_test
output_dir = data_dir + '/' + folder_test + '/' + output_name

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
#     # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
       print(e)


# --------------------------------------------------------------------------------
#                              glob test data path
# --------------------------------------------------------------------------------
img_path = glob.glob(test_images_path + '/*.tiff')
img_path.sort()


if not img_path:
    flag_recon = 1
    img_path = glob.glob(test_images_path + '/*')
    img_path.sort()
    # print(img_path)
    
    n_channel = len(glob.glob(img_path[0] + '/*.tiff'))
    # print(glob.glob(img_path[0]))
    # print(n_channel)

    output_dir = output_dir + 'SIM'


else:
    flag_recon = 1
    img_path = glob.glob(test_images_path)
    img_path.sort()
    n_channel = 9
    output_dir = output_dir + 'SIM'


if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# --------------------------------------------------------------------------------
#                          select models and load weights
# --------------------------------------------------------------------------------
modelFns = {'resu': resunet_up.att_unet32}
modelFN = modelFns[model_name]
optimizer = adam_v2.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
m = modelFN((input_height, input_width, n_channel))
m.summary()


m.load_weights(model_weights)
m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
im_count = 0
# val_time = []



for curp in img_path:
    # start_time = time()
    if flag_recon:
        imgfile = glob.glob(curp + '/*.tiff')
        imgfile.sort()
        img_batch = []
        for curf in imgfile:
            img = np.array(imageio.imread(curf).astype(np.float))
            img_batch.append(img)
        img = np.array(img_batch).transpose((1, 2, 0))
        img = img[np.newaxis, :, :, :]
    else:
        img = np.array(imageio.imread(curp).astype(np.float))
        img = img[np.newaxis, :, :, np.newaxis]

    img = prctile_norm(img)
    start_time = time()
    # print(start_time)
    pr = rm_outliers(prctile_norm(np.squeeze(m.predict(img, verbose=0))))
    end_time = time()
    # print(end_time)
    print('time:', end_time - start_time)
    # val_time.append(end_time - start_time)
    outName = curp.replace(test_images_path, output_dir)
    if not outName[-4:] == '.tiff':
        outName = outName + '.tif'
    img = Image.fromarray(np.uint16(pr * 65535))
    im_count = im_count + 1
    img.save(outName)

    # end_time = time()
    # print('time:', end_time - start_time)
    # val_time.append(end_time - start_time)

# print('val_time:', np.mean(val_time[-4:]))




