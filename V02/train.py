from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import sys
from image_distortions import *
import time
import os
import shutil

testIdx = 1 # 0,1

os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % testIdx


# --------------------------------------------------
print('setup parameters')
# --------------------------------------------------

ntrain = 4000
nvalid = 500
ntest = 715
imsize = 256
imcropsize = 128
nchannels = 1
vsize = imsize*imsize # vector size
batchsize = 128 # ntrain should be >= 2*batchsize

nsteps = 5000 # n training steps; set to 0 and restorevariables (below) to true if you just want to perform test
restorevariables = True # set to false to train from scratch; true to pickup training from last run

impath = '/home/mc457/Workspace/ImageForensics/SynthExamples' # where's the data
modelpath = '/home/mc457/Workspace/ImageForensics/TrainedModel/model%d.ckpt' % testIdx # where to save model
trainlogpath = '/home/mc457/Workspace/TFLog/ImageForensics%d/Train' % testIdx # where to save train log for tensorboard
validlogpath = '/home/mc457/Workspace/TFLog/ImageForensics%d/Valid' % testIdx # where to save validation log for tensorboard
testimoutpath = '/home/mc457/Workspace/ImageForensics/TestSynth' # where to save test images output results
testplotoutpath = '/home/mc457/Workspace/ImageForensics' # where to save plot with summary of test results

# see Models.py for model hyperparameters setup

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
    # return imgeomdeformandcrop(im)

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

def imgeomdeformandcrop(im): # only geometric deformations
    r = np.random.rand()

    if r < 0.9:
        im1 = imrandreflection(im)                                 if np.random.rand() < 0.5 else im
        im2 = imrandpptf(im1,drange)                               if np.random.rand() < 0.5 else im1
        im3 = imrandtform(im2,rotrange,sclrange,tlxrange,tlyrange) if np.random.rand() < 0.5 else im2
    else:
        im3 = im
    im4 = imcrop(im3)
    im4 = im4-np.min(im4)

    return im4

def contrdeformandcrop(im): # controlled deform and crop
    jm = np.copy(im)

    # deform
    # jm = imrandreflection(jm)
    # jm = imrandpptf(jm,drange)
    # jm = imrandtform(jm,rotrange,sclrange,tlxrange,tlyrange)

    # crop
    jm = imcrop(jm)
    jm = jm-np.min(jm)

    # adjust
    # jm = imrandjpegcompress(jm,qrange)
    # jm = imrandgammaadj(jm,gammarange)
    # jm = imrandlocaledit(jm)

    return jm


Valid = np.zeros((nvalid,imsize,imsize,nchannels))
Test = np.zeros((ntest,imsize,imsize,nchannels))

ivalid = -1
itest = -1
perm = np.arange(ntrain+nvalid+ntest)
np.random.shuffle(perm)
train_paths = []
for isample in range(0, ntrain):
    path = '%s/I%05d.png' % (impath,perm[isample]+1)
    train_paths.append(path)
for isample in range(ntrain, ntrain+nvalid):
    path = '%s/I%05d.png' % (impath,perm[isample]+1)
    im = misc.imread(path).astype(float)/255
    ivalid += 1
    Valid[ivalid,:,:,0] = im
for isample in range(ntrain+nvalid, ntrain+nvalid+ntest):
    path = '%s/I%05d.png' % (impath,perm[isample]+1)
    im = misc.imread(path).astype(float)/255
    itest += 1
    Test[itest,:,:,0] = im

ind_pairs_train = []
for i in range(nsteps):
    perm = np.arange(ntrain)
    np.random.shuffle(perm)
    for j in range(batchsize):
        ind_pairs_train.append([perm[j], perm[-j-1]])

def input_parser(pair):
    im0 = misc.imread(train_paths[pair[0]]).astype(float)/255
    im1 = misc.imread(train_paths[pair[1]]).astype(float)/255
    im = np.zeros((imcropsize,imcropsize,3))
    im[:,:,0] = imrandclutter(imcrop(im0))
    im[:,:,1] = imrandclutter(imdeformandcrop(im0))
    im[:,:,2] = imrandclutter(imdeformandcrop(im1))
    # im[:,:,1] = imrandclutter(contrdeformandcrop(im0))
    # im[:,:,2] = imrandclutter(contrdeformandcrop(im1))
    return im

train_pairs = tf.constant(ind_pairs_train)
tr_data = tf.data.Dataset.from_tensor_slices(train_pairs)
# tr_data = tr_data.map(lambda pair: tf.py_func(input_parser,[pair],tf.double))
tr_data = tr_data.map(lambda pair: tf.py_func(input_parser,[pair],tf.double), num_parallel_calls=12)
tr_data = tr_data.batch(batchsize)
tr_data = tr_data.prefetch(batchsize)

iterator = tf.data.Iterator.from_structure(tr_data.output_types,tr_data.output_shapes)
next_element = iterator.get_next()

tr_init_op = iterator.make_initializer(tr_data)
im1, im2, im3 = tf.split(next_element,3,3)
triplet_batch = tf.tuple((im1,im2,im3))

# --------------------------------------------------
print('model')
# --------------------------------------------------

from Models import Model
model = Model(nchannels, imcropsize, testIdx)
print('reslearn: ', model.residualLearning)

# --------------------------------------------------
print('train')
# --------------------------------------------------        

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

if os.path.exists(trainlogpath):
    shutil.rmtree(trainlogpath)
if os.path.exists(validlogpath):
    shutil.rmtree(validlogpath)
train_writer = tf.summary.FileWriter(trainlogpath, sess.graph)
valid_writer = tf.summary.FileWriter(validlogpath, sess.graph)

if restorevariables:
    saver.restore(sess, modelpath)
    print("Model restored.")
else:
    sess.run(tf.global_variables_initializer()) # do not use when restoring variables

batch_data1_Valid = np.zeros((nvalid//2,imcropsize,imcropsize,nchannels))
batch_data2_Valid = np.zeros((nvalid//2,imcropsize,imcropsize,nchannels))
batch_data3_Valid = np.zeros((nvalid//2,imcropsize,imcropsize,nchannels))

# validation set
print('setting up validation set')
for i in range(nvalid//2):
    batch_data1_Valid[i,:,:,0] = imrandclutter(imcrop(Valid[i,:,:,0]))
    batch_data2_Valid[i,:,:,0] = imrandclutter(imdeformandcrop(Valid[i,:,:,0]))
    batch_data3_Valid[i,:,:,0] = imrandclutter(imdeformandcrop(Valid[-i-1,:,:,0]))
    # batch_data2_Valid[i,:,:,0] = imrandclutter(contrdeformandcrop(Valid[i,:,:,0]))
    # batch_data3_Valid[i,:,:,0] = imrandclutter(contrdeformandcrop(Valid[-i-1,:,:,0]))

# train
print('training')

sess.run(tr_init_op)
best_triplet_loss = np.inf
count_no_improv = 0
count = 0
while True:
    try:
        batch_data1, batch_data2, batch_data3 = sess.run(triplet_batch)
        count = count+1
        print('step %d' % count)
    except tf.errors.OutOfRangeError:
        print('Done training.')
        break

    if count > 0 and count % 10 == 0:
        model.assign_running_averages(sess)

        summary, tl, acrc_valid = model.valid_step_with_summary(sess,batch_data1_Valid,batch_data2_Valid,batch_data3_Valid,nvalid)
        valid_writer.add_summary(summary, count)
        print('\ttl: %f' % tl)

        summary, acrc_batch, _ = model.train_step_with_summary(sess,batch_data1,batch_data2,batch_data3,2*batchsize)
        train_writer.add_summary(summary, count)

        print('\tacrc_batch: %f' % acrc_batch)
        print('\tacrc_valid: %f' % acrc_valid)

        if tl < best_triplet_loss:
            print("Saving model. Unsafe to CRTL+C.")
            save_path = saver.save(sess, modelpath)
            print("Model saved in file: %s" % save_path)

            best_triplet_loss = tl
            count_no_improv = 0
        else:
            count_no_improv += 1

        if count_no_improv == 1000:
            break # done with learning
    else:
        # t0 = time.time()
        model.train_step_without_summary(sess,batch_data1,batch_data2,batch_data3)
        # print(time.time()-t0)

train_writer.close()
valid_writer.close()

# --------------------------------------------------
print('test')
# --------------------------------------------------

hts = int(ntest/2)

batch_data1 = np.zeros((1,imcropsize,imcropsize,nchannels))
batch_data2 = np.zeros((1,imcropsize,imcropsize,nchannels))

acc = 0
plt.clf()

plt.plot([0, hts-1],[0.5, 0.5],'-k')

for i in range(hts):
    im1 = imrandclutter(imcrop(Test[i,:,:,0]))
    im2 = imrandclutter(imdeformandcrop(Test[i,:,:,0]))
    im3 = imrandclutter(imdeformandcrop(Test[-i-1,:,:,0]))
    # im2 = imrandclutter(contrdeformandcrop(Test[i,:,:,0]))
    # im3 = imrandclutter(contrdeformandcrop(Test[-i-1,:,:,0]))

    batch_data1[0,:,:,0] = im1
    test_label = 0
    if np.random.rand() < 0.5:
        batch_data2[0,:,:,0] = im2
    else:
        test_label = 1
        batch_data2[0,:,:,0] = im3

    sms = model.test(sess,batch_data1, batch_data2)
    smd = model.test2(sess,batch_data1, batch_data2)
    sms = (sms+smd)/2

    if test_label == 0:
        plt.plot(i,sms,'og')
    else:
        plt.plot(i,sms,'or')

    lbl = 0
    if sms > 0.5:
        s = 'Same'
    else:
        s = 'Diff'
        lbl = 1

    concat1 = np.concatenate((batch_data1[0,:,:,0],0.5*np.ones((imcropsize,5))),axis=1)
    concat2 = np.concatenate((concat1,batch_data2[0,:,:,0]),axis=1)
    
    if lbl == test_label:
        acc += 1
        path = '%s/T%05d_%s.png' % (testimoutpath,i,s)
    else:
        path = '%s/T%05d_%s_ERROR.png' % (testimoutpath,i,s)

    misc.imsave(path,concat2)

acrc = (float(acc)/float(hts))
print('accuracy: %f' % acrc)
plt.draw()
plt.savefig('%s/SgmdTest_Acrc%dPcnt.png' % (testplotoutpath,int(100*acrc)), bbox_inches='tight')

# --------------------------------------------------
print('clean up')
# --------------------------------------------------

sess.close()