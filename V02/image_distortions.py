from skimage import transform as trfm
from skimage import exposure as xpsr
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import string
import io

def imrotate(im,angle): # in degrees, with respect to center
    return trfm.rotate(im,angle)

def imrescale(im,factor): # with respect to center
    im2 = trfm.rescale(im,factor,mode='constant')
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
    return trfm.warp(im,tform,mode='constant')

def imsltf(im,angle,factor,tx,ty):
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

    return np.double(imout)

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
    return np.double(np.array(img))


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
    return np.double(np.array(img))

# erases a random circular area
def imranderase(im):
    im2 = np.copy(im)
    nr,nc = im2.shape
    r0 = np.random.randint(0.1*nr,0.9*nr)
    c0 = np.random.randint(0.1*nc,0.9*nc)
    d0 = np.random.randint(0.05*nr,0.2*nr)
    d1 = d0+10
    for i in range(nr):
        for j in range(nc):
            d = np.sqrt((i-r0)**2+(j-c0)**2)
            if d < d0:
                im2[i,j] = 0
            elif d < d1:
                im2[i,j] = (d-d0)/(d1-d0)*im2[i,j]
    return im2

# randomly picks a circular area and multiplies the pixel values by a random value in [1.5, 2.0]
def imrandlocalcontrast(im):
    im2 = np.copy(im)
    nr,nc = im2.shape
    r0 = np.random.randint(0.1*nr,0.9*nr)
    c0 = np.random.randint(0.1*nc,0.9*nc)
    d0 = np.random.randint(0.05*nr,0.2*nr)
    d1 = 2*d0
    f = 1.5+0.5*np.random.rand()
    for i in range(nr):
        for j in range(nc):
            d = np.sqrt((i-r0)**2+(j-c0)**2)
            if d < d0:
                im2[i,j] *= f
            elif d < d1:
                ff = (d-d0)/(d1-d0)
                im2[i,j] = (1-ff)*f*im2[i,j]+ff*im2[i,j]
    nim = im2
    mim = np.amax(im2)
    if mim > 0:
        nim = im2/mim
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

def imjpegcompress(im,quality):
    im8 = (255*im).astype(np.uint8)
    img = Image.fromarray(im8)
    string_buffer = io.BytesIO()
    img.save(string_buffer, format='JPEG', quality=quality)
    img = Image.open(string_buffer)
    return np.double(np.array(img))/255

def imrandjpegcompress(im,qrange):
    quality = np.random.randint(qrange[0],qrange[1])
    return imjpegcompress(im,quality)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy import misc
    
    def imshow(I):
        plt.imshow(I,cmap='gray')
        plt.axis('off')
        plt.show()

    def imshowlist(L):
        n = len(L)
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(L[i],cmap='gray')
            plt.axis('off')
        plt.show()

    impath = '/home/mc457/Workspace/ImageForensics/SynthExamples' # where's the data

    # random similarity transform parameters
    rotrange = [-45,45]
    sclrange = [50,150] # in percent
    tlxrange = [-20,20]
    tlyrange = [-20,20]
    # random perspective transform parameter
    drange = 20
    # random histogram transform parameter
    gammarange = [75,175] # actual gamma is this/100
    # jpeg compression parameter
    qrange = [10, 50]

    imsize = 256
    imcropsize = 128

    row0 = int((imsize-imcropsize)/2)
    col0 = row0

    def imcrop(im):
        return im[row0:row0+imcropsize,col0:col0+imcropsize]

    def imdeformandcrop(im):
        # return imcrop(xpsr.adjust_gamma(im,2.0))
        return imcrop(imjpegcompress(im,50))

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


    for i in range(100):
        path = '%s/I%05d.png' % (impath,1+10*i)
        I = np.double(misc.imread(path))/255
        J = imdeformandcrop(I)
        print(i,I.dtype,J.dtype)
        misc.imsave('/home/mc457/Workspace/ImageForensics/Scratch/I%03d.png' % i,np.uint8(255*np.concatenate((imcrop(I),J),axis=1)))