import cv2
import numpy as np

#SizeDilate = 1
#SizeErode = 1
#cv2.namedWindow("Erosion")
#cv2.namedWindow("Dilation")

SizeOuverture = 1
SizeFermeture = 1
cv2.namedWindow("Fermeture")
cv2.namedWindow("Ouverture")

img = cv2.imread("C:\\Users\\Toshiba\\Documents\\MASTER 2\\OCR\\data\\Page3.png",0)
#img = cv2.resize(img, (750, 750))
cv2.threshold(img,130,255,0,img)
img = cv2.bitwise_not(img)



def Closing():
    kernel = np.ones((SizeFermeture, SizeFermeture),np.uint8)
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow("Fermeture",img_closed)

def Opening():
    kernel = np.ones((SizeOuverture, SizeOuverture), np.uint8)
    img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("Ouverture", img_opened)

def ChangeFermSize(x):
    global SizeFermeture
    SizeFermeture = x
    Closing()

def ChangeOuverSize(x):
    global SizeOuverture
    SizeOuverture = x
    Opening()

def dilate_func():
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SizeDilate*2+1,SizeDilate*2+1))
    img_dilate = cv2.dilate(img,kernel,iterations=1)
    cv2.imshow("Dilation",img_dilate)

def erode_func():
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (SizeErode*2+1, SizeErode*2+1))
    img_erode = cv2.erode(img, kernel, iterations = 1)
    cv2.imshow("Erosion", img_erode)

def ChangeErosionSize(x):
    global SizeErode
    SizeErode = x
    erode_func()

def ChangeDilationSize(x):
    global SizeDilate
    SizeDilate = x
    dilate_func()

#cv2.createTrackbar("Size Erode","Erosion",1,21,ChangeErosionSize)
#cv2.createTrackbar("Size Dilate","Dilation",1,21,ChangeDilationSize)
cv2.createTrackbar("Ouverture Size","Ouverture",1,21,ChangeOuverSize)
cv2.createTrackbar("Fermeture Size","Fermeture",1,21,ChangeFermSize)

#erode_func()
#dilate_func()
Closing()
Opening()

cv2.waitKey(0)
cv2.destroyAllWindows()

