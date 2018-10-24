# Marcelo Cicconet, Oct 2017

from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import sys
from image_distortions import *
import time

with tf.device('/gpu:0'):
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
    batchsize = 256 # ntrain should be >= 2*batchsize

    nsteps = 1000 # n training steps; set to 0 and restorevariables (below) to true if you just want to perform test
    restorevariables = True # set to false to train from scratch; true to pickup training from last run

    impath = '/home/mc457/Images/SynthExamplesSelectedNoPlotsShuffle' # where's the data
    modelpath = '/home/mc457/Workspace/TFModel/SiamRelease.ckpt' # where to save model
    trainlogpath = '/home/mc457/Workspace/TFLog/SiamRelease/Train' # where to save train log for tensorboard
    validlogpath = '/home/mc457/Workspace/TFLog/SiamRelease/Valid' # where to save validation log for tensorboard
    testimoutpath = '/home/mc457/Workspace/Test' # where to save test images output results
    testplotoutpath = '/home/mc457/Workspace' # where to save plot with summary of test results

    # see Models.py for model hyperparameters setup

    # --------------------------------------------------
    print('setup data')
    # --------------------------------------------------

    # random similarity transform parameters
    rotrange = [-45,45]
    sclrange = [75,125] # in percent
    tlxrange = [-20,20]
    tlyrange = [-20,20]
    # random perspective transform parameter
    drange = 20
    # random histogram transform parameter
    gammarange = [75,125] # actual gamma is this/100

    row0 = int((imsize-imcropsize)/2)
    col0 = row0

    def imcrop(im):
        return im[row0:row0+imcropsize,col0:col0+imcropsize]

    def imdeformandcrop(im):
        # imout = imrandtform(imrandpptf(imrandreflection(im),drange),rotrange,sclrange,tlxrange,tlyrange)
        # imoutcrop = imout[row0:row0+imcropsize,col0:col0+imcropsize]
        # return imrandlocaledit(imrandgammaadj(imoutcrop,gammarange))

        r = np.random.rand()

        if r < 0.9:
            im1 = imrandreflection(im)                                 if np.random.rand() < 0.5 else im
            im2 = imrandpptf(im1,drange)                               if np.random.rand() < 0.5 else im1
            im3 = imrandtform(im2,rotrange,sclrange,tlxrange,tlyrange) if np.random.rand() < 0.5 else im2
        else:
            im3 = im
        im4 = im3[row0:row0+imcropsize,col0:col0+imcropsize]
        im4 = im4-np.min(im4)
        if r < 0.9:
            im5 = imrandgammaadj(im4,gammarange)                       if np.random.rand() < 0.5 else im4
            im6 = imrandlocaledit(im5)                                 if np.random.rand() < 0.5 else im5
        else:
            im6 = im4
        return im6
    
    Train = np.zeros((ntrain,imsize,imsize,nchannels))
    Valid = np.zeros((nvalid,imsize,imsize,nchannels))
    Test = np.zeros((ntest,imsize,imsize,nchannels))

    itrain = -1
    ivalid = -1
    itest = -1
    perm = np.arange(ntrain+nvalid+ntest)
    np.random.shuffle(perm)
    for isample in range(0, ntrain):
        path = '%s/I%05d.png' % (impath,perm[isample]+1)
        im = misc.imread(path).astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
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

    # --------------------------------------------------
    print('model')
    # --------------------------------------------------

    from Models import Model
    model = Model(nchannels, imcropsize)

    # --------------------------------------------------
    print('train')
    # --------------------------------------------------        

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

    train_writer = tf.summary.FileWriter(trainlogpath, sess.graph)
    valid_writer = tf.summary.FileWriter(validlogpath, sess.graph)

    if restorevariables:
        saver.restore(sess, modelpath)
        print("Model restored.")
    else:
        sess.run(tf.global_variables_initializer()) # do not use when restoring variables

    batch_data1 = np.zeros((batchsize,imcropsize,imcropsize,nchannels))
    batch_data2 = np.zeros((batchsize,imcropsize,imcropsize,nchannels))
    batch_data3 = np.zeros((batchsize,imcropsize,imcropsize,nchannels))

    batch_data1_Valid = np.zeros((nvalid/2,imcropsize,imcropsize,nchannels))
    batch_data2_Valid = np.zeros((nvalid/2,imcropsize,imcropsize,nchannels))
    batch_data3_Valid = np.zeros((nvalid/2,imcropsize,imcropsize,nchannels))

    # validation set
    for i in range(nvalid/2):
        batch_data1_Valid[i,:,:,0] = imrandclutter(imcrop(Valid[i,:,:,0]))
        batch_data2_Valid[i,:,:,0] = imrandclutter(imdeformandcrop(Valid[i,:,:,0]))
        batch_data3_Valid[i,:,:,0] = imrandclutter(imdeformandcrop(Valid[-i-1,:,:,0]))

    # train
    best_triplet_loss = np.inf
    count_no_improv = 0
    for i in range(nsteps+1):
        print('step %d' % i)

        perm = np.arange(ntrain)
        np.random.shuffle(perm)

        for j in range(batchsize):
            batch_data1[j,:,:,0] = imrandclutter(imcrop(Train[perm[j],:,:,0]))
            batch_data2[j,:,:,0] = imrandclutter(imdeformandcrop(Train[perm[j],:,:,0]))
            batch_data3[j,:,:,0] = imrandclutter(imdeformandcrop(Train[perm[-j-1],:,:,0]))

        if i % 10 == 0:
            model.assign_running_averages(sess)

            summary, tl, acrc_valid = model.valid_step_with_summary(sess,batch_data1_Valid,batch_data2_Valid,batch_data3_Valid,nvalid)
            valid_writer.add_summary(summary, i)
            print('\ttl: %f' % tl)

            summary, acrc_batch, _ = model.train_step_with_summary(sess,batch_data1,batch_data2,batch_data3,2*batchsize)
            train_writer.add_summary(summary, i)

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
            model.train_step_without_summary(sess,batch_data1,batch_data2,batch_data3)

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
    for i in range(hts):
        im1 = imrandclutter(imcrop(Test[i,:,:,0]))
        im2 = imrandclutter(imdeformandcrop(Test[i,:,:,0]))
        im3 = imrandclutter(imdeformandcrop(Test[-i-1,:,:,0]))

        batch_data1[0,:,:,0] = im1
        test_label = 0
        if np.random.rand() < 0.5:
            batch_data2[0,:,:,0] = im2
        else:
            test_label = 1
            batch_data2[0,:,:,0] = im3

        sms = model.test(sess,batch_data1, batch_data2)

        plt.plot(i,0.5,'.k')
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