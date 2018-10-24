# Marcelo Cicconet, Jan 2017

# http://scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html

from skimage import transform as trfm
from skimage import exposure as xpsr
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import string

def imrotate(im,angle): # in degrees, with respect to center
    return trfm.rotate(im,angle)

def imrescale(im,factor): # with respect to center
    im2 = trfm.rescale(im,factor)
    [w1,h1] = im.shape
    [w2,h2] = im2.shape
    r1 = int(h1/2)
    c1 = int(w1/2)
    r2 = int(h2/2)
    c2 = int(w2/2)
    if w2 > w1:
        imout = im2[r2-int(h1/2):r2-int(h1/2)+h1,c2-int(w1/2):c2-int(w1/2)+w1]
    else:
        imout = np.zeros((h1,w1))
        imout[r1-int(h2/2):r1-int(h2/2)+h2,c1-int(w2/2):c1-int(w2/2)+w2] = im2
    return imout

def imtranslate(im,tx,ty): # tx: columns, ty: rows
    tform = trfm.SimilarityTransform(translation = (-tx,-ty))
    return trfm.warp(im,tform)

def imsltf(im,angle,factor,tx,ty):
    # im = imrotate(im,angle)
    # print(im.shape)
    # im = imrescale(im,factor)
    # print(im.shape)
    # im = imtranslate(im,tx,ty)
    # print(im.shape)
    return imtranslate(imrescale(imrotate(im,angle),factor),tx,ty)

# random similarity transform
def imrandtform(im,rotrange,sclrange,tlxrange,tlyrange):
    randag = np.random.randint(rotrange[0],rotrange[1])
    randsc = np.random.randint(sclrange[0],sclrange[1])
    randtx = np.random.randint(tlxrange[0],tlxrange[1])
    randty = np.random.randint(tlyrange[0],tlyrange[1])
    return imsltf(im,randag,float(randsc)/100.0,randtx,randty)

# http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

# random perspective transform
# ***it's recommended to do perspective transform before similarity transform*** (due to issue with PIL)
def imrandpptf(im,drange):
    img = Image.fromarray(im)
    width, height = img.size

    d = np.random.randint(-drange,drange,8)
    dtl = d[0:2] # x, y
    dtr = d[2:4]
    dbr = d[4:6]
    dbl = d[6:8]

    coeffs = find_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)], # destination points (top-left, top-right, bottom-right, bottom-left)
        [(dtl[0], dtl[1]), (width+dtr[0], dtr[1]), (width+dbr[0], height+dbr[1]), (dbl[0], height+dbl[1])]) # original points

    img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    imout = np.array(img)
    mx = np.amax(imout)
    if mx > 1:
        imout = imout/mx

    return imout

def imrandreflection(im):
    imout = im
    if np.random.rand() < 0.5:
        imout = np.fliplr(imout)
    if np.random.rand() < 0.5:
        imout = np.flipud(imout)
    return imout

# random gamma adjustment
def imrandgammaadj(im,gammarange):
    gamma = float(np.random.randint(gammarange[0],gammarange[1]))/100.0
    return xpsr.adjust_gamma(im,gamma)

fonts = ['FreeMonoBoldOblique', 'FreeSansBoldOblique', 'FreeSerifBoldItalic',
         'FreeMonoBold',        'FreeSansBold',        'FreeSerifBold',
         'FreeMonoOblique',     'FreeSansOblique',     'FreeSerifItalic',
         'FreeMono',            'FreeSans',            'FreeSerif']

def imrandtext(im):
    img = Image.fromarray(im)
    draw = ImageDraw.Draw(img)
    randfont = fonts[np.random.randint(0,8)]
    randsize = np.random.randint(16,32)
    width, height = img.size
    randplacement = [np.random.randint(0,width/2), np.random.randint(0,height)]
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/%s.ttf' % (randfont), 16)
    nletters = len(string.ascii_letters)
    l = np.random.randint(5,15)
    s = ''
    for i in range(l):
        s = s+string.ascii_letters[np.random.randint(0,nletters)]
    draw.text((randplacement[0], randplacement[1]),'%s' % (s),0.75+0.25*np.random.rand(),font=font)
    return np.array(img)


def imrandrect(im):
    img = Image.fromarray(im)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    randplacement = [np.random.randint(0,width/2), np.random.randint(0,height/2)]
    randsize = [np.random.randint(width/3,width/2), np.random.randint(height/3,height/2)]
    w = np.random.randint(1,3)
    f = np.random.rand()
    draw.line([(randplacement[0],randplacement[1]),(randplacement[0],randplacement[1]+randsize[1])],width=w,fill = f)
    draw.line([(randplacement[0],randplacement[1]),(randplacement[0]+randsize[0],randplacement[1])],width=w,fill = f)
    draw.line([(randplacement[0]+randsize[0],randplacement[1]),(randplacement[0]+randsize[0],randplacement[1]+randsize[1])],width=w,fill = f)
    draw.line([(randplacement[0],randplacement[1]+randsize[1]),(randplacement[0]+randsize[0],randplacement[1]+randsize[1])],width=w,fill = f)
    return np.array(img)

# erases a random circular area
def imranderase(im):
    nr,nc = im.shape
    r0 = np.random.randint(0.1*nr,0.9*nr)
    c0 = np.random.randint(0.1*nc,0.9*nc)
    d0 = np.random.randint(0.05*nr,0.2*nr)
    d1 = d0+10
    for i in range(nr):
        for j in range(nc):
            d = np.sqrt((i-r0)**2+(j-c0)**2)
            if d < d0:
                im[i,j] = 0
            elif d < d1:
                im[i,j] = (d-d0)/(d1-d0)*im[i,j]
    return im

# randomly picks a circular area and multiplies the pixel values by a random value in [1.5, 2.0]
def imrandlocalcontrast(im):
    nr,nc = im.shape
    r0 = np.random.randint(0.1*nr,0.9*nr)
    c0 = np.random.randint(0.1*nc,0.9*nc)
    d0 = np.random.randint(0.05*nr,0.2*nr)
    d1 = 2*d0
    f = 1.5+0.5*np.random.rand()
    for i in range(nr):
        for j in range(nc):
            d = np.sqrt((i-r0)**2+(j-c0)**2)
            if d < d0:
                im[i,j] *= f
            elif d < d1:
                ff = (d-d0)/(d1-d0)
                im[i,j] = (1-ff)*f*im[i,j]+ff*im[i,j]
    nim = im
    mim = np.amax(im)
    if mim > 0:
        nim = im/mim
    return nim

# replaces a random circular area with another random circular area froom the same image
def imrandreplace(im):
    nr,nc = im.shape
    r0 = np.random.randint(0.3*nr,0.7*nr,(2))
    c0 = np.random.randint(0.3*nc,0.7*nc,(2))
    d0 = np.random.randint(0.05*nr,0.15*nr)
    d1 = 1.5*d0
    imout = np.zeros(im.shape)
    for i in range(nr):
        for j in range(nc):
            d = np.sqrt((i-r0[0])**2+(j-c0[0])**2)
            if d < d0:
                imout[i,j] = im[i-r0[0]+r0[1],j-c0[0]+c0[1]]
            elif d < d1:
                ff = (d-d0)/(d1-d0)
                imout[i,j] = (1-ff)*im[i-r0[0]+r0[1],j-c0[0]+c0[1]]+ff*im[i,j]
            else:
                imout[i,j] = im[i,j]
    return imout

def imrandclutter(im): # network should be able to distinguish equal/different images independently of this
    nprr = np.random.rand()
    if nprr < 0.25:
        imout = imrandrect(im)
    elif nprr < 0.5:
        imout = imrandtext(im)
    else:
        imout = im
    return imout

def imrandlocaledit(im): # to be applied to images the network is supposed to detect as equivalent
    nprr = np.random.rand()
    if nprr < 0.5:
        imout = imranderase(im)
    else:
        imout = imrandlocalcontrast(im)
    return imout