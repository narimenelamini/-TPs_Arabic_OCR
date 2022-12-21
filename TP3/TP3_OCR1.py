import cv2
import numpy as np

img = cv2.imread("C:\\Users\\Toshiba\\Documents\\MASTER 2\\OCR\\data\\texte1.jpg", cv2.IMREAD_GRAYSCALE)
bw = np.zeros(img.shape,np.uint8)
th = 0
type = 0
def afficher():
    cv2.threshold(img,th,255,type,bw)
    cv2.imshow("result",bw)

def ChangeTh(x):
    global th
    th = x
    afficher()

def ChangeType(x):
    global type
    type = x
    afficher()

cv2.namedWindow("result")
cv2.createTrackbar("threshold", "result",0,255,ChangeTh)
cv2.createTrackbar("type", "result",0,4,ChangeType)

afficher()
cv2.waitKey(0)
cv2.destroyAllWindows()