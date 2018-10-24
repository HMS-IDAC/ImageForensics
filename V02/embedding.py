# Marcelo Cicconet, Sep 2017

from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import sys
from image_distortions import *
import time
from tensorflow.contrib.tensorboard.plugins import projector
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'


# --------------------------------------------------
print('setup parameters')
# --------------------------------------------------

ntest = 32*32
imsize = 256
imcropsize = 128 # should be the same as in train.py
nchannels = 1 # should be the same as in train.py
vsize = imsize*imsize # vector size, should be the same as in train.py
batchsize = 256

impath = '/home/mc457/Workspace/ImageForensics/SynthExamples'
modelpath = '/home/mc457/Workspace/ImageForensics/TrainedModel/model1.ckpt'
embedlogpath = '/home/mc457/Workspace/ImageForensics/Embedding'

# to see embedding run 'tensorboard --logdir=___' replacing ___ with embedlogpath path, and click on EMBEDDINGS

# --------------------------------------------------
print('setup data')
# --------------------------------------------------

# random similarity transform parameters
rotrange = [-45,45]
sclrange = [50,150] # in percent # [75,125]
tlxrange = [-20,20]
tlyrange = [-20,20]
# random perspective transform parameter
drange = 20
# random histogram transform parameter
gammarange = [50,200] # [75,175]; actual gamma is this/100
# jpeg compression parameter
qrange = [10, 50]

row0 = int((imsize-imcropsize)/2)
col0 = row0

def imcrop(im):
    return im[row0:row0+imcropsize,col0:col0+imcropsize]

def imdeformandcrop(im):
    r = np.random.rand()

    if r < 0.9:
        im1 = imrandreflection(im)                                 if np.random.rand() < 0.5 else im
        im2 = imrandpptf(im1,drange)                               if np.random.rand() < 0.5 else im1
        im3 = imrandtform(im2,rotrange,sclrange,tlxrange,tlyrange) if np.random.rand() < 0.5 else im2
    else:
        im3 = im
    im4 = imcrop(im3)
    im4 = im4-np.min(im4)
    if r < 0.9:
        im5 = imrandjpegcompress(im4,qrange)                       if np.random.rand() < 0.5 else im4
        im6 = imrandgammaadj(im5,gammarange)                       if np.random.rand() < 0.5 else im5
        im7 = imrandlocaledit(im6)                                 if np.random.rand() < 0.5 else im6
    else:
        im7 = im4

    return im7


Test = np.zeros((ntest,imsize,imsize,nchannels))

for isample in range(ntest):
    path = '%s/I%05d.png' % (impath,isample+1)
    im = misc.imread(path).astype(float)/255
    Test[isample,:,:,0] = im

# --------------------------------------------------
print('model')
# --------------------------------------------------

from Models import Model
model = Model(nchannels, imcropsize, 1)

# --------------------------------------------------
print('recover parameters')
# --------------------------------------------------        

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

saver.restore(sess, modelpath)
print("Model restored.")

# --------------------------------------------------
print('embedding')
# --------------------------------------------------        

batch_sprite = np.zeros((ntest,imcropsize,imcropsize,nchannels))
for i in range(ntest):
    # batch_sprite[i,:,:,0] = imrandclutter(imcrop(Test[i,:,:,0]))
    batch_sprite[i,:,:,0] = imrandclutter(imdeformandcrop(Test[i,:,:,0]))

cnn0 = sess.run(model.cnn_test[0],feed_dict={model.tf_data[0]: batch_sprite})

embedding_var = tf.Variable(cnn0, name='cnn0')
sess.run(embedding_var.initializer)
summary_writer = tf.summary.FileWriter(embedlogpath)
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

n = int(np.sqrt(ntest))
sprite = np.zeros((n*imcropsize,n*imcropsize))
for i in range(n):
    for j in range(n):
        I = batch_sprite[i*n+j,:,:,0]
        sprite[i*imcropsize:(i+1)*imcropsize,j*imcropsize:(j+1)*imcropsize] = I
sprite_path = '%s/%s' % (embedlogpath,'sprite.png')
misc.imsave(sprite_path, sprite)

embedding.sprite.image_path = sprite_path
embedding.sprite.single_image_dim.extend([imcropsize, imcropsize])

projector.visualize_embeddings(summary_writer, config)
saver = tf.train.Saver([embedding_var])
saver.save(sess, '%s/%s' % (embedlogpath,'test.ckpt'))

# --------------------------------------------------
print('clean up')
# --------------------------------------------------

sess.close()