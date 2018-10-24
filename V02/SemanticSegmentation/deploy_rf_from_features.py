import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys
from ToolBox.imtools import *
from ToolBox.ftools import *
from skimage.exposure import equalize_adapthist as adapthisteq

def resizeFM(I,nr,nc):
    F = np.zeros((nr,nc,I.shape[2]))
    for i in range(I.shape[2]):
        fm = I[:,:,i]
        mx = np.max(np.abs(fm))
        if mx > 0:
            fm = fm/mx
        fm = mx*imresizeDouble(fm,[nr,nc])
        F[:,:,i] = fm
    return F


annotationsPath = '/home/mc457/Workspace/ImageForensics/ImagesForSegmentation'
featuresPath = '/home/mc457/Workspace/ImageForensics/FeaturesForSegmentation'
rfModelPath = '/home/mc457/Workspace/ImageForensics/TrainedModelRF/rfc.data'


rfc = loadData(rfModelPath)


# classify

nImages = len(listfiles(annotationsPath,'_Data.tif'))
nClasses = 2
nFeatures = 32+64+128

for i in range(nImages):
    print(i)

    I = tifread(pathjoin(annotationsPath,'I%03d_Data.tif' % (i+1)))
    A = tifread(pathjoin(annotationsPath,'I%03d_Labels.tif' % (i+1))) == 1
    nr = I.shape[0]
    nc = I.shape[1]

    # fPath = pathjoin(featuresPath,'I%03d_Features.tif' % (i+1))
    # Features = tifread(fPath)
    # F = resizeFM(Features,nr,nc)

    F = np.zeros((nr,nc,nFeatures))

    imPath = pathjoin(featuresPath,'I%03d_Features1.tif' % (i+1))
    I = tifread(imPath)
    F1 = resizeFM(I,nr,nc)

    imPath = pathjoin(featuresPath,'I%03d_Features2.tif' % (i+1))
    I = tifread(imPath)
    F2 = resizeFM(I,nr,nc)

    imPath = pathjoin(featuresPath,'I%03d_Features3.tif' % (i+1))
    I = tifread(imPath)
    F3 = resizeFM(I,nr,nc)

    F[:,:,:32] = F1
    F[:,:,32:32+64] = F2
    F[:,:,32+64:] = F3

    # M = np.zeros((nr,nc,nClasses))
    # out = rfc.predict(F.reshape(-1,nFeatures))
    # C = out.reshape((nr,nc))
    # for i in range(nClasses):
    #     M[:,:,i] = C == i+1

    # N = np.zeros((nr,nc,nClasses))
    # out = rfc.predict_proba(F.reshape(-1,nFeatures))
    # for i in range(nClasses):
    #     N[:,:,i] = out[:,i].reshape((nr,nc))

    out = rfc.predict_proba(F.reshape(-1,nFeatures))
    tifwrite(np.uint8(255*out[:,0].reshape((nr,nc))),pathjoin(annotationsPath,'I%03d_PredictionRF.tif' % (i+1)))
    # imshowlist([adapthisteq(I),A,M[:,:,0],N[:,:,0]])