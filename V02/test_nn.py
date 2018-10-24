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
impathout = '/home/mc457/Images/ImageForensics/SynthExamples/TestNN'

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
# classes.sort()

impaths = []
for c in classes:
    impaths.append(os.listdir('%s/%s' % (impath,c)))

nclasses = len(impaths)

Branch0 = np.zeros((nclasses,imcropsize,imcropsize,nchannels))
Branch1 = np.zeros((nclasses,imcropsize,imcropsize,nchannels))

nn = 5
# P = np.zeros(((nn+1)*128,nclasses*128))
count = 0
for i in range(nclasses):
    P = np.zeros(((nn+1)*128,128))

    imI = '%s/%s/%s' % (impath,classes[i],impaths[i][0])
    n1 = len(impaths[i])
    imI2 = '%s/%s/%s' % (impath,classes[i],impaths[i][np.random.randint(1,n1)])
    I = misc.imread(imI).astype(float)/255
    I2 = misc.imread(imI2).astype(float)/255


    imgs = []
    for j in range(nclasses):
        if j == i:
            J = I2
        else:
            n2 = len(impaths[j])
            imJ = '%s/%s/%s' % (impath,classes[j],impaths[j][np.random.randint(0,n2)])
            J = misc.imread(imJ).astype(float)/255

        imgs.append(J)
        # imshowlist([I,J])
        Branch0[j,:,:,0] = I
        Branch1[j,:,:,0] = J


        # imshowlist([I,J])

    p = 0.5*(model.test(sess,Branch0,Branch1)+model.test2(sess,Branch0,Branch1))
    idx = np.argsort(p,0)


    # P[:128,128*i:128*(i+1)] = I
    P[:128,:] = I
    for j in range(nn):
        I = imgs[int(idx[-1-j,0])]
        # P[(j+1)*128:(j+2)*128,128*i:128*(i+1)] = I
        P[(j+1)*128:(j+2)*128,:] = I

    tifwrite(np.uint8(255*P),pathjoin(impathout,'I%05d.tif' % i))

    for j in range(nn):
        nni = int(idx[-1-j,0])
        if nni == i:
            count += 1
            break

    print(i,nni)

sess.close()

print('acc: ', float(count)/nclasses)
# imwrite(np.uint8(255*P),'/home/mc457/Workspace/NN.tif')
imshow(P)