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

'''#Appels
inter = 10
theta, maxi = get_angle_rot(bw,inter)
print('theta = ' , theta)
print('maxi = ' , maxi)
im_result = rotation(bw, theta)

#Affichage
cv2.imshow('initial', ~bw)
cv2.imshow('des', ~im_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#sauvegarde des resultats
if not(isdir('fixed')):
    mkdir('fixed')

cv2.imwrite('fixed/rotated_projection.png',~bw )'''
#======Redressement méthode de traitement des rectangles minimum des parties connexes================
#Calcul angle de rotation
def get_angle_rotation(im):
    coord_s = np.column_stack(np.where(im > 0))
    angle = cv2.minAreaRect(coord_s)[-1]
    if angle < -45 :
        angle = -(90 + angle)
    else:
        angle = -angle
    return  angle

'''#Appels
inter = 10
theta = get_angle_rotation(bw)
print('theta = ' , theta)
im_result = rotation(bw, theta)
des = rotation(bw ,-theta)
print('theta = ' , theta)

#Affichage
cv2.imshow('initial', ~bw)
cv2.imshow('des', ~des)
cv2.waitKey(0)
cv2.destroyAllWindows()

#sauvegarde des resultats
if not(isdir('fixed_im')):
    mkdir('fixed_im')

cv2.imwrite('fixed_im/rotated_projection.png',~des )'''

#===========================================Méthode de Hough==============================

plt.figure(figsize=(8,6))
plt.imshow(img, cmap = plt.cm.gray)
plt.axis('off')
bw = cv2.threshold(img,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )[1]


#definition du contours
def contours(img):
    h,w = img.shape
    im = np.zeros(img.shape , np.uint8)

    for i in range(h-1):
        for j in range(w-1):
            if(img[i,j] != img[i, j+1]) or (img[i,j] != img[i + 1, j]):
                im[i,j] = 255
    return im

img1 = contours(bw)
plt.figure(figsize = (8,6))
plt.imshow(img1, cmap = plt.cm.gray)
plt.axis('off')

lines = cv2.HoughLines(img1, rho = 1 , theta = np.pi/180, threshold=300)
nbl = len(lines)
print('nbl', nbl)

 #affichage des résultats
fixed = rotation(img, -theta, 1)
plt.figure(figsize=(8,6))
plt.imshow(fixed, cmap = plt.cm.gray)
plt.axis('off')













