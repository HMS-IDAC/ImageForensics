import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys
from ToolBox.imtools import *
from ToolBox.ftools import *


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

nClasses = 2
nImages = len(listfiles(annotationsPath,'_Data.tif'))
nFeatures = 16+32+64+128


# parse label folder (assumes class balance)


imList = []
lbList = []
nSamples = 0
for iImage in range(nImages):
    print(iImage)

    I0 = tifread(pathjoin(annotationsPath,'I%03d_Data.tif' % (iImage+1)))
    nr = I0.shape[0]
    nc = I0.shape[1]

    F = np.zeros((nr,nc,nFeatures))

    imPath = pathjoin(featuresPath,'I%03d_Features0.tif' % (iImage+1))
    I = tifread(imPath)
    F0 = resizeFM(I,nr,nc)

    imPath = pathjoin(featuresPath,'I%03d_Features1.tif' % (iImage+1))
    I = tifread(imPath)
    F1 = resizeFM(I,nr,nc)

    imPath = pathjoin(featuresPath,'I%03d_Features2.tif' % (iImage+1))
    I = tifread(imPath)
    F2 = resizeFM(I,nr,nc)

    imPath = pathjoin(featuresPath,'I%03d_Features3.tif' % (iImage+1))
    I = tifread(imPath)
    F3 = resizeFM(I,nr,nc)

    F[:,:,:16] = F0
    F[:,:,16:16+32] = F1
    F[:,:,16+32:16+32+64] = F2
    F[:,:,16+32+64:] = F3

    # F[:,:,:32] = F1
    # F[:,:,32:32+64] = F2
    # F[:,:,32+64:] = F3

    # F[:,:,:64] = F2
    # F[:,:,64:] = F3

    # F = F3
    
    lbPath = pathjoin(annotationsPath,'I%03d_Labels.tif' % (iImage+1))
    A = tifread(lbPath)

    # imshowlist([I0,F[:,:,0],A == 1,A == 2])
    # sys.exit(0)

    imList.append(F)

    L = []
    for iClass in range(nClasses):
        Li = (A == (iClass+1))*(np.random.rand(A.shape[0],A.shape[1]) < 0.5) # using only x% to train faster
        L.append(Li)
        nSamples += np.sum(L[iClass] > 0)
    lbList.append(L)



# setup training


X = np.zeros((nSamples,nFeatures))
Y = np.zeros((nSamples))
i0 = 0
for iImage in range(len(imList)):
    print(iImage)
    F = imList[iImage]
    for iClass in range(nClasses):
        Li = lbList[iImage][iClass]
        indices = Li > 0
        l = np.sum(indices)
        x = np.zeros((l,nFeatures))
        for iFeat in range(nFeatures):
            Fi = F[:,:,iFeat]
            xi = Fi[indices]
            x[:,iFeat] = xi
        y = (iClass+1)*np.ones((l))
        X[i0:i0+l,:] = x
        Y[i0:i0+l] = y
        i0 = i0+l



# train

rfc = RandomForestClassifier(n_estimators=20,n_jobs=-1,min_samples_leaf=60)
rfc = rfc.fit(X, Y)



# plot feat importance


fi = rfc.feature_importances_
fn = []
for i in range(nFeatures):
    fn.append('F%03d' % i)
plt.rcdefaults()
fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
y_pos = range(len(fn))
ax.barh(y_pos, fi)
ax.set_yticks(y_pos)
ax.set_yticklabels(fn)
ax.invert_yaxis()
ax.set_title('Feature Importance')
plt.show()


saveData(rfc,rfModelPath)


# classify

def classify(idx):
    F = imList[idx]
    sI = [F.shape[0],F.shape[1]]

    M = np.zeros((sI[0],sI[1],nClasses))
    out = rfc.predict(F.reshape(-1,nFeatures))
    C = out.reshape((sI[0],sI[1]))
    for i in range(nClasses):
        M[:,:,i] = C == i+1

    N = np.zeros((sI[0],sI[1],nClasses))
    out = rfc.predict_proba(F.reshape(-1,nFeatures))
    for i in range(nClasses):
        N[:,:,i] = out[:,i].reshape((sI[0],sI[1]))

    I = tifread(pathjoin(annotationsPath,'I%03d_Data.tif' % (idx+1)))
    A = tifread(pathjoin(annotationsPath,'I%03d_Labels.tif' % (idx+1))) == 1

    return I, A, M, N

# sys.exit(0)

for i in range(nImages):
    print(i)
    I, A, M, N = classify(i)

    out = cat(1,normalize(I),N[:,:,0])
    out = cat(1,out,A)

    tifwrite(np.uint16(65535*out),'/home/mc457/Workspace/ImageForensics/ProbMaps/PM%03d.tif' % i)