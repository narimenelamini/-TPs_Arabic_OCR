'''                      
                                    TP1
                                   =====
            Histogram projection & Text Segmentation to PAWS-lines
'''


import cv2
import numpy as np
from matplotlib import pyplot as plt
from genericpath import isdir
from operator import is_not
from os import mkdir

from sklearn.cluster import DBSCAN
#  =====================PART ONE : Historgam projections=========================
#Reading image
img = cv2.imread("data/Page3.png")
print(img.shape)
# Convert the image to gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#binarization 
bw = cv2.threshold(img,130,255,cv2.THRESH_BINARY)[1]
#Inversing the image
bw = cv2.bitwise_not(bw)

 #We can binarize and inverse the image in one call** 
#bw = cv2.threshold(img,130,255,cv2.THRESH_BINARY_INV)[1]

#Horizontal histogram projection
def histo_h(im):
    h = im.shape[0]
    hist_h = np.zeros((h), np.uint16)
    for i in range(h):
        hist_h[i] = np.count_nonzero(im[i,:])
    return hist_h
#Vertical histogram projection
def histo_v(im):
    w = im.shape[1]
    hist_v = np.zeros((w), np.uint16)
    for i in range(w):
        hist_v[i] = np.count_nonzero(im[:,i])
    return hist_v

hist_h = histo_h(bw)
hist_v = histo_v(bw)

 #Initialize two white spaces with the same image shape to display the hist projection
im_hist_h = np.zeros(img.shape,np.uint8)
im_hist_h[:,:] = 255
im_hist_v = np.zeros(img.shape,np.uint8)
im_hist_v[:,:] = 255

#Putting 0 (black) on the image alight pixels
h,w = img.shape
for i in range(0,h):
    im_hist_h[i,0:hist_h[i]] = 0
for i in range(0,w):
    im_hist_v[0:hist_v[i],i] = 0
# Rep creation
if not(isdir('result')):
    mkdir('result')
if not(isdir('res')):
    mkdir('res')
#Saving images with appropriate histograms
cv2.imwrite('result/hist_h.png', cv2.hconcat([img,im_hist_h]))
cv2.imwrite('result/hist_v.png', cv2.vconcat([img, im_hist_v]))

#Displaying results
cv2.imwrite('horiz.png', im_hist_h)
cv2.imshow('histVert', cv2.hconcat([img,im_hist_h]))

#  =====================PART TOW : Lines detection  from the horizontal histo====================

def lines_detection(im,vh):
    lines_pos = []
    lines = []
    k = 0
    h = len(vh)
    beg = 0
    end = 0
    i = 0
    while(i<h-1):
        if((vh[i] == 0) & (vh[i+1] > 0)):
            k +=1
            beg = i
            for j in range(beg,h-1):
                if((vh[j] > 0) & (vh[j+1]== 0)):

                    end = j
                    lines_pos.append([beg,end])
                    line = im[beg:end,0:]
                    lines.append(line)
                    break
                else:
                    j+=1
            i = end+1
        else:
            i+=1
    return  k, lines_pos,lines
his_h = histo_h(bw)
nbl,pos_lgs,lgs = lines_detection(bw, hist_h)
print('nb_lines', nbl)
print('loc', pos_lgs)
h,w = lgs[0].shape
print(h,w)
#Saving the image
for i in range(nbl):
    cv2.imwrite('res/lig'+str(i)+'.png', lgs[i])
    

for i in range(nbl):
    hist_h1 = histo_h(lgs[i])
    hist_v = histo_v(lgs[i])
    im_hist_h = np.zeros(lgs[i].shape,np.uint8)
    im_hist_h[:,:] = 255
    im_hist_v = np.zeros(lgs[i].shape,np.uint8)
    im_hist_v[:,:] = 255
    cv2.imshow('vertHist', cv2.hconcat([~lgs[i], im_hist_h]))
    cv2.imwrite('result/his_h'+str(i)+'.png', cv2.hconcat([~lgs[i], im_hist_h]))
    cv2.imshow('horisHist', cv2.hconcat([~lgs[i], im_hist_v]))
    cv2.imwrite('result/his_v'+str(i)+'.png', cv2.hconcat([~lgs[i], im_hist_v]))
    cv2.waitKey(0)
cv2.destroyAllWindows()


lg_base = np.zeros(nbl)
for i in range(nbl):
    hist_h1 = histo_h(lgs[i])
    lg_base[i] = np.argmax(hist_h1)
def calcul_bd_base(lb,fraction):
    long = (lb [1] - lb[0])* fraction/100
    return long

bd_base = np.zeros(nbl)
for i in range(nbl):
    bd_base[i] = calcul_bd_base(pos_lgs[i],20)

plt.figure(figsize = (8,6))
for i in range(1, nbl+1):
    plt.subplot(4,1,i)
    plt.imshow(~lgs[i-1], cmap=plt.cm.gray)
    plt.plot([0,w], [lg_base[i-1], lg_base[i-1]], label='lg_base')
    plt.plot([0,w], [lg_base[i-1]- bd_base[i-1], lg_base[i-1]+ bd_base[i-1]], label='bd_inf')
    plt.title('Line'+str(i))
    plt.axis('off')
plt.legend()
plt.savefig('res.png')
plt.show()


#  =====================PART THREE : PAWs detection & seperation=========================
import os
path = 'res'
files = os.listdir(path)
img = []
for i in range(len(files)):
    img1 = cv2.imread(('res/'+files[i]),0)
    img.append(img1)
    cv2.imshow(str(i), img[i])
    cv2.imwrite('result/line'+str(i)+'.png', img[i])
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('res/lig3.png')
print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
bw = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
def detection_paws(im,vh):
    pos_paws = []
    paws = []
    k= 1
    h = len(vh)
    beg = 0
    end = 0
    i = 0
    j=0
    while(i<h-1):
        if((vh[i] == 0) & (vh[i+1] >0)):
            k+=1
            beg = i+1
            for j in range(beg, h-1):
                if((vh[j]>0) & (vh[j+1] == 0)):
                    end = j+1
                    pos_paws.append([beg,end])
                    paw = im[0:beg:end]
                    paws.append(paw)
                    break
                else:
                    j+=1
            i=end
        else:
            j+=1
    return k+1 ,pos_paws,paw

nbp, pos_pws, pws = detection_paws(bw,hist_v)
print("nb_paws",nbp,'\t loc', pos_pws)
for i in range(nbp):
    print(i)
    cv2.imwrite('result/PAWS/_line3_'+str(i)+'.png',pws[i])


