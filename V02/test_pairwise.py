from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from image_distortions import *
import time
import os

import sys
from ToolBox.imtools import *
from ToolBox.ftools import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# --------------------------------------------------
print('parameters')
# --------------------------------------------------

imcropsize = 128 # should be the same as in train.py
nchannels = 1 # should be the same as in train.py
modelpath = '/home/mc457/Workspace/ImageForensics/TrainedModel/model.ckpt'
impath = '/home/mc457/Images/ImageForensics/SynthExamples/Test'

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
print('test')
# --------------------------------------------------

classes = os.listdir(impath)
classes.sort()

impaths = []
for c in classes:
    impaths.append(os.listdir('%s/%s' % (impath,c)))

nclasses = len(impaths)

Branch0 = np.zeros((nclasses,imcropsize,imcropsize,nchannels))
Branch1 = np.zeros((nclasses,imcropsize,imcropsize,nchannels))

P = np.zeros((nclasses,nclasses))
for i in range(nclasses):
    print(i)

    imI = '%s/%s/%s' % (impath,classes[i],impaths[i][0])
    n1 = len(impaths[i])
    imI2 = '%s/%s/%s' % (impath,classes[i],impaths[i][np.random.randint(1,n1)])
    I = misc.imread(imI).astype(float)/255
    I2 = misc.imread(imI2).astype(float)/255
    for j in range(nclasses):
        if j == i:
            J = I2
        else:
            n2 = len(impaths[j])
            imJ = '%s/%s/%s' % (impath,classes[j],impaths[j][np.random.randint(0,n2)])
            J = misc.imread(imJ).astype(float)/255

        # imshowlist([I,J])
        Branch0[j,:,:,0] = I
        Branch1[j,:,:,0] = J


        # imshowlist([I,J])

    p = 0.5*(model.test(sess,Branch0,Branch1)+model.test2(sess,Branch0,Branch1))
    # print(p)
    P[:,i] = p[:,0]


sess.close()

tifwrite(np.uint8(100*P),'/home/mc457/Workspace/SelfSimilarity.tif')

imshow(P)