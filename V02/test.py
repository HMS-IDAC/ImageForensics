from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import sys
from image_distortions import *
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# --------------------------------------------------
print('parameters')
# --------------------------------------------------

imcropsize = 128 # should be the same as in train.py
nchannels = 1 # should be the same as in train.py
modelpath = '/home/mc457/Workspace/ImageForensics/TrainedModel/model.ckpt'
impath = '/home/mc457/Images/ImageForensics/SynthExamples/Test'
outfigpath = '/home/mc457/Workspace/ImageForensics/AcrcTest.png'
nruns = 1
imtestoutpath = '/home/mc457/Workspace/ImageForensics/TestSynth' # images saved only when nruns == 1
thr = 0.5

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

impaths = []
for c in classes:
    impaths.append(os.listdir('%s/%s' % (impath,c)))

nclasses = len(impaths)

Anchor = np.zeros((nclasses,imcropsize,imcropsize,nchannels))
Same  = np.zeros((nclasses,imcropsize,imcropsize,nchannels))
Diff  = np.zeros((nclasses,imcropsize,imcropsize,nchannels))

_aa = np.zeros(nruns)
_as = np.zeros(nruns)
_ad = np.zeros(nruns)
acc_errors_same = np.zeros(nclasses)
acc_errors_diff = np.zeros(nclasses)

for irun in range(nruns):
    l_anchor = []
    l_same = []
    l_diff = []
    for i in range(nclasses):
        j = i
        while j == i:
            j = np.random.randint(0,nclasses)

        nsameclass = len(impaths[i])
        ndiffclass = len(impaths[j])

        perm = np.arange(nsameclass)
        np.random.shuffle(perm)

        anchor = '%s/%s/%s' % (impath,classes[i],impaths[i][perm[0]])
        # print('%d: %s' % (i,anchor))
        same = '%s/%s/%s' % (impath,classes[i],impaths[i][perm[1]])

        k = np.random.randint(0,ndiffclass)
        diff = '%s/%s/%s' % (impath,classes[j],impaths[j][k])

        l_anchor.append('%s_%s' % (classes[i],impaths[i][perm[0]]))
        l_same.append('%s_%s' % (classes[i],impaths[i][perm[1]]))
        l_diff.append('%s_%s' % (classes[j],impaths[j][k]))

        Anchor[i,:,:,0] = misc.imread(anchor).astype(float)/255
        Same[i,:,:,0]  = misc.imread(same).astype(float)/255
        Diff[i,:,:,0]  = misc.imread(diff).astype(float)/255

    sms = 0.5*(model.test(sess,Anchor,Same)+model.test2(sess,Anchor,Same))
    smd = 0.5*(model.test(sess,Anchor,Diff)+model.test2(sess,Anchor,Diff))
    # [sms,smd] = model.test_triplet(sess,Anchor,Same,Diff)

    acc = 0
    accS = 0
    accD = 0

    for i in range(nclasses):
        if sms[i] > thr:
            plt.plot(i,int(100*sms[i]),'sg',markersize=2)
            s = 'Same'
            acc += 1
            accS += 1
        else:
            acc_errors_same[i] += 1
            plt.plot(i,int(100*sms[i]),'sr',markersize=2)
            s = 'Diff_ERROR'

        if nruns == 1:
            concat1 = np.concatenate((Anchor[i,:,:,0],0.5*np.ones((imcropsize,5))),axis=1)
            concat2 = np.concatenate((concat1,Same[i,:,:,0]),axis=1)
            misc.imsave('%s/I_idx%03d_anc_cls%s_%03d_%s.png' % (imtestoutpath,i,l_anchor[i][:-4],100*sms[i],s),concat2)

        if smd[i] > thr:
            acc_errors_diff[i] += 1
            plt.plot(i,int(100*smd[i]),'or',markersize=2)
            s = 'Same_ERROR'
        else:
            plt.plot(i,int(100*smd[i]),'og',markersize=2)
            s = 'Diff'
            acc += 1
            accD += 1
        
        if nruns == 1:            
            concat1 = np.concatenate((Anchor[i,:,:,0],0.5*np.ones((imcropsize,5))),axis=1)
            concat2 = np.concatenate((concat1,Diff[i,:,:,0]),axis=1)
            misc.imsave('%s/I_idx%03d_anc_cls%s_%03d_%s.png' % (imtestoutpath,i,l_anchor[i][:-4],100*smd[i],s),concat2)

    acrc = (float(acc)/float(2*nclasses))
    acrcS = (float(accS)/float(nclasses))
    acrcD = (float(accD)/float(nclasses))
    _aa[irun] = acrc
    _as[irun] = acrcS
    _ad[irun] = acrcD

plt.plot([0, nclasses-1],[thr*100, thr*100],'-k')
plt.axis([-1,nclasses,-10,110])
plt.xlabel('class of first image in pair')
plt.ylabel('likelihood of images being the same')
plt.draw()
plt.savefig(outfigpath, bbox_inches='tight')

print('acrc aggregated: mean %f, std %f' % (np.mean(_aa),np.std(_aa)))
print('acrc same      : mean %f, std %f' % (np.mean(_as),np.std(_as)))
print('acrc diff      : mean %f, std %f' % (np.mean(_ad),np.std(_ad)))

# print('errors per class (class, true positive same, true positive diff')
# for i in range(nclasses):
#     print('%03d' % i, '%03d' % int(acc_errors_same[i]), '%03d' % int(acc_errors_diff[i]))

sess.close()