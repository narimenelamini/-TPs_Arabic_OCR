import cv2
import matplotlib.pyplot as plt

#======================================Zhang & Suen squelettization==============================

# lecture et binarisation

img = cv2.imread('')
im = cv2.threshold(img, 0,1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def voisins(im,x,y):
    voisinage = (im[x-1,y], im[x-1 , y+1], im[x, y+1], im[x+1, y+1], im[x+1,y], im[x+1,y-1], im[x, y-1], im[x-1 , y-1])
    return voisinage

def somme_voisins(v):
    return sum(v)

#Transition
def transitions(v):
    tr = 0
    for i in range(len(v) -1):
        if (v[i] == 0 and v[i+1] == 1):
            tr +=1
    if(v[-1] == 0 and v[0] == 1):
        tr +=1
    return tr

#Squelette

def ZS_squelette(img):
    im = img.copy()
    h,w = im.shape
    liste1 = 1
    liste2 = 1

    while(liste1  or liste2):
        liste1 = []
        for i in range(1, h-1):
            for j in range(1, w-1):
                if im[i,j] == 1:
                    v = voisins(im,i,j)
                    s = somme_voisins(v)
                    tr = transitions(v)
                    if(2 <= s <=6) and (tr == 1) and (v[0]*v[2]*v[4] == 0) and (v[2]*v[4]*v[6] == 0):
                        liste1.append([i,j])
        for l,s in liste1:
            im[l,s] = 0
            liste2 = []
        for i in range(1, h-1):
                for j in range(1, w-1):
                    if im[i,j] == 1:
                        v = voisins(im,i,j)
                        s = somme_voisins(v)
                        tr = transitions(v)
                        if(2 <= s <=6) and (tr == 1) and (v[0]*v[2]*v[4] == 0) and (v[2]*v[4]*v[6] == 0):
                            liste2.append([i,j])
        for k,m in liste2:
            im[k,m] = 0
    return im


#======================================Hilditch squelettization algorithm=========================

def Hilditch_squelette(img):
    im = img.copy()
    h,w = im.shape
    liste = 1
   

    while(liste):
        liste = []
        for i in range(1, h-1):
            for j in range(1, w-1):
                if im[i,j] == 1:
                    v = voisins(im,i,j)
                    v1 = voisins(im,i-1,j)
                    v2 = voisins(im,i,j+1)
                    s = somme_voisins(v)
                    tr = transitions(v)
                    tr1 = transitions()
                    tr2 = transitions(v)
                    if((2 <= s <=6) and (tr == 1) and (v[0]*v[2]*v[4] == 0) and (v[2]*v[4]*v[6] == 0 or tr1 != 1 ) and (v[0]*v[2]*v[4] == 0 or tr2!=1)):
                        liste.append([i,j])
        
        for l,s in liste:
            im[l,s] = 0
    return im
#====================================Affichage des résultats====================================#
#Appel
thinned  = Hilditch_squelette(im)
# Affichage
plt.figure()
plt.imshow(~im , cmap=plt.cm.gray)
plt.axis('off')
plt.title('original')
plt.figure()
plt.imshow(~thinned , cmap=plt.cm.gray)
plt.axis('off')
plt.title('squelette')

