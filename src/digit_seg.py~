import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

def farthest_neighbor_min_point(series):
    #find min point
    #first eliminate noise using convolution
    mins = []
    con_series = np.convolve(series, np.asarray([0.2,0.2,0.2,0.2,0.2]), mode='valid')
    for i in range(1, len(con_series)-1, 1):
        if(con_series[i-1]>con_series[i]) and (con_series[i]<con_series[i+1]):
            mins.append(i)
    #find the most isolated point
    mini = 0
    minjk = 0
    for i in mins:
        j = i-1
        while(con_series[j]>con_series[i]) and (j>0):
            j -= 1
        k = i+1
        while(con_series[k]>con_series[i]) and (k<len(con_series)-1):
            k += 1
        if j*k > minjk:
            minjk = j*k
            mini = i
    
    return i
    
def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    cv2.imshow("custom_blur_demo", dst)

#img = cv2.imread('../pics/num_whole/2019-03-18-16:47:38.jpg')
img = cv2.imread('/home/jacesamostawa/Downloads/timg.jpeg')
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(im.shape)
im2 = custom_blur_demo(im)
cv2.waitKey(0)

area = []
pos = []
sum0 = np.sum(im,0)

sum1 = np.sum(im,1)
y = np.arange(0,im.shape[0],1)
plt.figure(1)
plt.scatter(y, sum1, alpha=0.6)

#Find the turning point

cut = farthest_neighbor_min_point(sum1)
im = im[:cut, :]
sum0 = np.sum(im,0)
sum1 = np.sum(im,1)

cv2.imshow('new',im)
cv2.waitKey(0)

for i in range(im.shape[1]):
    area.append(sum0[i]/im.shape[0])
    s = 0
    for j in range(im.shape[0]):
        s += im[j,i]*j
    s = s/im.shape[0]
    pos.append(s)

x = np.arange(0,im.shape[1],1)
    
plt.figure(2)
plt.subplot(211)
plt.scatter(x,np.asarray(area), alpha=0.6)

plt.subplot(212)
plt.scatter(x,pos)

plt.show()

