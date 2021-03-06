import numpy as np
import cv2
from matplotlib import pyplot as plt 


img = cv2.imread('2019-03-18-15:49:01.jpg')
img = img[540:620,660:800]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('',img)
corners = cv2.goodFeaturesToTrack(gray,6,0.01,20)#棋盘上的所有点

corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)#在原图像上画出角点位置


cv2.imshow('p',img)
cv2.waitKey(0)

#plt.imshow(img)
#plt.show()

