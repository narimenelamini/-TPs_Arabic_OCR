
import cv2 
import numpy as np 
import matplotlib.pylab as plt

def binarization(image):
        newImage = image.copy()
        if newImage.shape[-1]==3:
            newImage = cv2.cvtColor(newImage,cv2.COLOR_BGR2GRAY)
        tresh = cv2.threshold(newImage,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        return cv2.bitwise_not(tresh)

def invert(image):
    return cv2.bitwise_not(image.copy())

def hist(im):
    h,w =im.shape
    hist_h = np.zeros((h),np.uint16)
    for j in range(0,h):
        for i in range(0,w):
            if im[j,i]==255:
                hist_h[j]+=1
    return hist_h

'''def getangle(image,theta):
    h,w = image.shape
    maxi = -9999
    ga = 0
    for angle in range(-theta,theta):
        print('testing: ',angle)
        ##Rotate
        
        M = cv2.getRotationMatrix2D((h//2,w//2),angle=angle,scale=1.0)
        newimage = cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_CUBIC)
        his = hist(newimage)
        maxh = max(his)
        print(angle,maxh)
        if maxh>maxi:
            maxi =maxh
            ga=angle
    return ga'''

def rotate(image):
    h,w = image.shape
    #an = getangle(image,10)
    an = -3
    M = cv2.getRotationMatrix2D((h//2,w//2),angle=an,scale=1.0)
    return  cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_CUBIC)
    


#Affichage
image = cv2.imread('data/Page3_rotated.png',0)
image = invert(image)
new = rotate(image)

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.title('Before')
plt.imshow(~image,cmap='gray')
fig.add_subplot(1,2,2)
plt.imshow(~new,cmap='gray')
plt.title('AFter')
plt.axis('OFF')
plt.show()