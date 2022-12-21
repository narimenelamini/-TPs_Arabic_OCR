import cv2
import numpy as np
from genericpath import isdir
from operator import is_not
from os import mkdir
from matplotlib import pyplot as plt
#=====================Redressement méthode de projection d’histogramme=============================
# mettre aux niveaux de gris
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#binariser
def binariser(image):
    return cv2.threshold(image, 0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]

#inverser
def inverser(image):
    return cv2.bitwise_not(image)
   
#lecture & appels

img = cv2.imread('data/rotated.png')
gray = get_grayscale(img)
thresh1 = binariser(gray)
bw = inverser(thresh1)

####### Rotation
def rotation(im, angle, scale = 1):
    (h,w) = im.shape[:2]
    center = (w/2 , h/2)
    #first, we calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center , angle, scale)
    #Now, we apply a wrapAffine with affine
    rotated = cv2.warpAffine(im, M , (w,h) , flags=cv2.INTER_CUBIC , borderMode=0 , borderValue=(0,0,0))
    return rotated


#projection histogram horizontal
def histo_h(im):
    h,w = im.shape
    hist_h = np.zeros((h) , np.uint16)
    for j in range(0,h):
        for i in range(0,w):
            if im[j,i] == 255 :
                hist_h[j] +=1
    return hist_h
# get the angle of rotation
def get_angle_rot(im, inter):
    maximum = np.zeros(2*inter+1)
    max = 0
    t_rot = 0
    for theta in range(-inter , inter+1):
        img_rot = rotation(im, theta,1 )
        hist = histo_h(img_rot)
        maxi = np.max(hist)
        if maxi>max :
            max = maxi
            t_rot = theta
        maximum[theta+inter] = maxi
        return t_rot,maximum

def get_angle_rotation(im):
    print(im.shape)
    coord_s = np.column_stack(np.where(im > 0))
    print(len(coord_s))
    angle = cv2.minAreaRect(coord_s)[-1]
    print("angle = ", angle)
    if angle < -45 :
        angle = -(90 + angle)
    else:
        angle = -angle
    return  angle
def contours(img):
    h,w = img.shape
    im = np.zeros(img.shape , np.uint8)

    for i in range(h-1):
        for j in range(w-1):
            if(img[i,j] != img[i, j+1]) or (img[i,j] != img[i + 1, j]):
                im[i,j] = 255
    return im

#Appels
img1 = contours(bw)
lines = cv2.HoughLines(img1, rho = 1 , theta = np.pi/180, threshold=300)
nbl = len(lines)
print('nbl', nbl)

#Affichage



fig = plt.figure()
fig.add_subplot(1,2,1)
plt.title('Before')
plt.imshow(~img1,cmap='gray')
plt.axis('OFF')
plt.show()