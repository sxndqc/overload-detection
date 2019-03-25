import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from math import *

def interpolate_expand(a,b):
    #b to a
    l = len(a)
    lb = len(b)
    y = np.zeros(l+1)
    step = l/lb
    for i in range(lb):
        r_left = step*i
        r_right = step*(i+1)
        if floor(r_left)==floor(r_right):
            y[floor(r_left)] += b[i]    
        else:
            y[floor(r_left)] += b[i]*(ceil(r_left)-r_left)/step
            y[floor(r_right)] += b[i]*(r_right - floor(r_right))/step
    y = y[:-1]
    return y

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

def find_several_min_points(series, maxo_mini, mini_max):
    #find a few min point
    #firstly eliminate noise using convolution
    #a minimum point must be near a point that surpasses a mini_max
    #a minimum turning point must be below a maxo_mini
    mins = []
    con_series = np.convolve(series, np.asarray([0.2,0.2,0.2,0.2,0.2]), mode='valid')
    valid_mins = set()
    
    
    for i in range(1, len(con_series)-1, 1):
        if(con_series[i-1]>con_series[i]) and (con_series[i]<con_series[i+1]) and (con_series[i] < maxo_mini):
            mins.append(i)
    #find the most isolated point
    
    for i in range(len(mins)-1):
        surpass = False;
        for p in series[mins[i]:mins[i+1]]:
            if p>mini_max:
                surpass = True;
                break
        if surpass:
            valid_mins.add(mins[i])
            valid_mins.add(mins[i+1])
            
    return list(valid_mins)


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    cv2.imshow("custom_blur_demo", dst)

if __name__=="__main__":
    #img = cv2.imread('../pics/num_whole/2019-03-18-16:47:38.jpg')
    img = cv2.imread('/home/jacesamostawa/wieght-limit-recgonize/pics/num_whole/2019-03-18-16:47:38.jpg')
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    print(im.shape)
    #im2 = custom_blur_demo(im)
    #cv2.waitKey(0)
    
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
    
    #cv2.imshow('new',im)
    #cv2.waitKey(0)
    
    for i in range(im.shape[1]):
        area.append(sum0[i]/im.shape[0])
        s = 0
        for j in range(im.shape[0]):
            s += im[j,i]*j**2
        s = s/im.shape[0]
        pos.append(s)
    
    x = np.arange(0,im.shape[1],1)
        
    plt.figure(2)
    plt.subplot(211)
    plt.scatter(x,np.asarray(area), alpha=0.6)
    
    plt.subplot(212)
    plt.scatter(x,pos)
    plt.show()
    
    valid_mins = find_several_min_points(area, 60, 100)
    valid_mins.sort()
    nums = []
    print(valid_mins)
    for i in range(len(valid_mins)-1):
        #nums.append(cv2.resize(cv2.copyMakeBorder(im[:,valid_mins[i]:valid_mins[i+1]],0,0, (cut-valid_mins[i+1] + valid_mins[i])//  2, (cut-valid_mins[i+1] + valid_mins[i])//2, cv2.BORDER_REPLICATE), (28, 28)))
        #cv2.imshow(str(i),nums[i])
        nums.append(im[:,valid_mins[i]:valid_mins[i+1]])
        #cv2.imwrite("/home/jacesamostawa/mnist-competition/"+str(i)+".jpg",nums[i])
        
    #cv2.waitKey(0)
    """
    from model import *
    from vgg16 import *
    from vgg5 import *
    model1 = VGGNet("model/vggnet.h5")
    model2 = VGGNet5("model/vggnet5.h5")
    
    def b(X):
        return np.reshape(X, [28,28])
    def f(X):
        return np.reshape(X, [-1,28,28,1])
    def s(X):
        cv2.imshow('',b(X))
        cv2.waitKey(0)
    
    m1 = []
    m2 = []
    
    for num in nums:
        ret,thresh = cv2.threshold(b(num),127,255,cv2.THRESH_BINARY)
        kernel = np.ones((2,2), np.uint8)
        erode = cv2.erode(b(thresh), kernel, iterations=1)
        
        m1.append(np.argmax(model1.predict(f(erode))))
        m2.append(np.argmax(model2.predict(f(erode))))
        
        s(erode)
    
    print(m1)
    print(m2)
    """
    # Using waves
    """
    ls = pkl.load(open("ls.pkl","rb"))
    us = pkl.load(open("us.pkl","rb"))
    
    #plt.figure(1)
    numnum = [4,9,7,4,0]
    
    for k,num in enumerate(nums):
        #ret, im = cv2.threshold(num,127,255,cv2.THRESH_BINARY)
        im = num
        area = [];
        pos = []
        
        sum0 = np.sum(im,0)
        sum1 = np.sum(im,1)
    
        #cv2.imshow('new',im)
        #cv2.waitKey(0)
    
        for i in range(im.shape[1]):
            area.append(sum0[i]/im.shape[0])
            s = 0
            for j in range(im.shape[0]):
                s += im[j,i]*j**2
            s = s/im.shape[0]
            pos.append(s)
        
        area = np.asarray(area)
        
        dist = np.fft.fft(area, 1000)
        
        x = np.arange(0,len(area),1)
        
        plt.figure(k)
        plt.subplot(211)
        plt.scatter(x,area)
        plt.grid(True)
        plt.subplot(212)
        plt.scatter(x,interpolate_expand(pos, us[numnum[k]]))
        plt.grid(True)
        plt.show()
    
        #print(np.correlate(dist[:500], dls[0][:500]))
        #print(np.sum(dist[:500]**2))
        #print(np.sum(dls[0][:500]**2))
                
        #print(  np.correlate(dist[:500],dist[:500])/np.sqrt(np.sum(dist[:500]**2)*np.sum(dist[:500]**2)))
        cor = [np.corrcoef(area, interpolate_expand(pos,i))[0,1] for i in us]
        predict = np.argmax(cor)
        print(cor)
        print(predict)
        
    plt.show()
    """
    #using similarity
    ims = pkl.load(open("ims.pkl",'rb'))
    kernel = np.ones((10,10),np.uint8)
    
    for i,num in enumerate(nums):
        ret, num = cv2.threshold(num,127,255,cv2.THRESH_BINARY)
        kernel = np.ones((10, 10), np.uint8)
        num = cv2.morphologyEx(num, cv2.MORPH_OPEN, kernel)
        width = np.nonzero(np.sum(num,0)) 
        height = np.nonzero(np.sum(num,1))
        num = num[height[0][0]:height[0][-1], width[0][0]:width[0][-1]]
        
        minr = 1e10
        k = -1
        for j,im in enumerate(ims):
            im = cv2.resize(im, (num.shape[1], num.shape[0]))
            im = cv2.dilate(im, kernel, iterations=1)
            #8之所以识别错是因为旁边还有小颗粒没有去除掉
            #还是没解决
            
            
            #print(num.shape)
            #print(im.shape)
            #not only dilate, but have to locate the white;
            #num = np.asarray(num)
            #im = np.asarray(im)
            nn = np.array(num, dtype=float)
            ii = np.array(im, dtype = float)  #把类型设置成float是关键！否则总是按照Uint来计算，没法abs
            #test = np.zeros(ii.shape)
            #test[20:50,:] = 255
            #cv2.imshow('',np.abs(nn - test))
            #cv2.waitKey(0)
            
            #不知道为什么永远无法把im中的白色部分减成白色
            
            #cv2.imshow('',np.abs(num-im))
            #cv2.waitKey(0)
            residue = np.abs(nn - ii)
            #print(residue)
            r = np.sum(residue,(0,1))
            if r<minr:
                minr = r
                k = j
                #cv2.imshow('',residue)  
                #cv2.imshow('',im)
                #cv2.waitKey(0)
                #cv2.imshow('',im)
                #cv2.waitKey(0)
            #cv2.imshow('',residue)
        print(k)
        #cv2.waitKey(0)
         
   
        
    
