from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import sys
from image_distortions import *
import time
import os

import sys
sys.path.insert(0, '/home/mc457/Documents/Python/ImageScience')
from toolbox.imtools import *
from toolbox.ftools import *

os.environ['CUDA_VISIBLE_DEVICES']='0'

# --------------------------------------------------
print('parameters')
# --------------------------------------------------

imcropsize = 128 # should be the same as in train.py
nchannels = 1 # should be the same as in train.py
modelpath = '/home/mc457/Workspace/ImageForensics/TrainedModel/model.ckpt'

# --------------------------------------------------
print('load model and parameters')
# --------------------------------------------------

from Models import Model
model = Model(nchannels, imcropsize, 1)

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

saver.restore(sess, modelpath)
print("Model restored.")

# --------------------------------------------------
print('deploy')
# --------------------------------------------------


dirPath = '/home/mc457/Workspace/ImageForensics/ImagesForSegmentation'
dirPathOut = '/home/mc457/Workspace/ImageForensics/FeaturesForSegmentation'

nImages = len(listfiles(dirPath,'_Data.tif'))
for i in range(nImages):
    print(i)

    filePath = pathjoin(dirPath,'I%03d_Data.tif' % (i+1))
    I = im2double(tifread(filePath))

    Batch = I.reshape((1,I.shape[0],I.shape[1],1))

    F = np.squeeze(sess.run(model.b0l3_test,feed_dict={model.tf_data[0]: Batch}))
    tifwrite(F,pathjoin(dirPathOut,'I%03d_Features3.tif' % (i+1)))

    F = np.squeeze(sess.run(model.b0l2_test,feed_dict={model.tf_data[0]: Batch}))
    tifwrite(F,pathjoin(dirPathOut,'I%03d_Features2.tif' % (i+1)))

    F = np.squeeze(sess.run(model.b0l1_test,feed_dict={model.tf_data[0]: Batch}))
    tifwrite(F,pathjoin(dirPathOut,'I%03d_Features1.tif' % (i+1)))

    F = np.squeeze(sess.run(model.b0l0_test,feed_dict={model.tf_data[0]: Batch}))
    tifwrite(F,pathjoin(dirPathOut,'I%03d_Features0.tif' % (i+1)))


# --------------------------------------------------
print('cleanup')
# --------------------------------------------------

sess.close()