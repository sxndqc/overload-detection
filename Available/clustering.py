import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
import matplotlib.image as mpimg
import cv2

def valid_approx(approx):
    if len(approx)!=4:
        return False
    approx = sorted(approx, key=lambda x: x[0][0])
    #只能选长边，短边有透视变化，斜率差距大
    #print(approx)
    #print("emm: ",approx[2]," ", approx[2][0]," ", approx[2][0][1])
    k1 = (approx[2][0][1] - approx[0][0][1]).astype(np.float32) / (approx[2][0][0] - approx[0][0][0]).astype(np.float32)
    k2 = (approx[3][0][1] - approx[1][0][1]).astype(np.float32) / (approx[3][0][0] - approx[1][0][0]).astype(np.float32)
    #print("k1: %f, k2: %f"%(k1, k2))
    
    if np.abs(k1-k2) > 1e-1:
        return False
    #print("valid")
    return True
    
def extractValid(cnt, img_t):
    a = cv2.contourArea(cnt, True)
    #print(a)
    approx = np.array([False])
    if 1350 <= np.abs(a) <= 1750:
        epsilon = 0.05*cv2.arcLength(cnt,True)
        tapprox = cv2.approxPolyDP(cnt,epsilon,True)
        if valid_approx(tapprox):
            approx = tapprox
            cv2.drawContours(img_t,[approx],-1,(0,225,0),1) 
        #print(approx)
        hull = cv2.convexHull(cnt)
        #print(hull)
        '''
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        '''
        
            #cv2.polylines(img_r,[hull],True,(0,0,255),1)
    return approx

def get_perspective_mat(src_points):
    src_points = sorted(src_points, key=lambda x: x[0][0])
    src_points = np.array(src_points, dtype = "float32")
    dst_points = np.array([[85., 80.],  [85., 60.], [140.,80.], [140., 60.]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return M

def cut(image, M):
    rows,cols = image.shape[:2]
    res = cv2.warpPerspective(image, M, (rows, cols), cv2.INTER_LINEAR)
    im = cv2.resize(res[60:80, 100:140],(600,200))
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("PT_resize", im)  
    #cv2.waitKey(0)
    return im
    #return im


def ROIExtraction(img):
    img = cv2.resize(img,(1920,1080))
    img = img[500:700,600:800,:3]

    img1 = np.reshape(img, (img.shape[0]*img.shape[1], 3))
    img1.flags.writeable = True
    cl = cluster.KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

    d = cl.fit(img1)

    y = d.predict(img1)

    #print(y)
    #print("???")
    
    for i in range(len(y)):
        if y[i] == 0:
            img1[i,0] = 255
            img1[i,1] = 0
            img1[i,2] = 0

        if y[i] == 1:
            img1[i,1] = 255
            img1[i,0] = 0
            img1[i,2] = 0

        if y[i] == 2:
            img1[i,2] = 255
            img1[i,1] = 0
            img1[i,0] = 0

        if y[i] == 3:
            img1[i,0] = 255
            img1[i,1] = 255
            img1[i,2] = 255


    #approx = None
    img2 = np.reshape(img1,(img.shape[0], img.shape[1], 3))
    #print(img2)
    cv2.imshow("filtered", img2)  
    cv2.waitKey(6)
    #print("!!!")
    img_r = img.copy()
    img_g = img.copy()
    img_b = img.copy()
    r,g,b = cv2.split(img2)

    ret, binary = cv2.threshold(r,127,255,cv2.THRESH_BINARY)
    
    _,contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    #print(contours)
    #cv2.drawContours(img_r,contours,-1,(0,225,0),1) 
    #cv2.drawContours(r,contours,-1,(50,50,225),1)
    #cv2.imshow("r", img_r)  
    #cv2.waitKey(0)
    
    for cnt in contours:
        tapprox = extractValid(cnt, img_r)
        if tapprox.any():
            approx = tapprox
            
    #cv2.imshow("r", img_r)  
    #cv2.waitKey(0)

    ret, binary = cv2.threshold(g,127,255,cv2.THRESH_BINARY) 
    
    _,contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    #cv2.drawContours(img_g,contours,-1,(0,255,0),1)
    #cv2.imshow("g", img_g)  
    #cv2.waitKey(0) 
    
    for cnt in contours:
        tapprox = extractValid(cnt, img_g)
        if tapprox.any():
            approx = tapprox
    #cv2.imshow("g", img_g)  
    #cv2.waitKey(0)  

    ret, binary = cv2.threshold(b,127,255,cv2.THRESH_BINARY) 
    
    _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img_b,contours,-1,(0,255,0),1)
    #cv2.imshow("b", img_b)  
    #cv2.waitKey(0)
    #print(contours) 
    for cnt in contours:
        tapporox = extractValid(cnt, img_b)
        if tapprox.any():
            #print(tapprox)
            approx = tapprox

    #cv2.imshow("b", img_b)  
    #cv2.waitKey(0)  

    print(approx)
    #Video 里面根本找不出来任何区域
    #rr = cv2.circle(img_b, approx, 1,'b')
    #cv2.imshow("points", rr)  
    #cv2.waitKey(0)
    M = get_perspective_mat(approx)
    im = cut(img, M)
    return im
    
    
    
if __name__=="__main__":
    img = cv2.imread("2019-03-18-16:07:57.jpg")
    NotFound = True
    while NotFound:
        try:
            result = ROIExtraction(img)
            NotFound = False
        except:
            NotFound = True
    
    cv2.imshow("result", result)  
    cv2.waitKey(6)
'''


edges = cv2.Sobel(img,cv2.CV_16S,1,1) 

edges = cv2.convertScaleAbs(edges) 
plt.figure()
plt.imshow(edges,plt.cm.gray) 


edgesh = cv2.Sobel(img,cv2.CV_16S,1,0) 
edgesh = cv2.convertScaleAbs(edgesh) 
plt.figure()
plt.imshow(edgesh,plt.cm.gray) 


edgesv = cv2.Sobel(img,cv2.CV_16S,0,1) 
edgesv = cv2.convertScaleAbs(edgesv) 
plt.figure()
plt.imshow(edgesv,plt.cm.gray)

后面的四个数字threshold可以降低
然后凸四边形限定
要区分遮挡和光线干扰
模板的叠加
自动化数字位置的问题可能需要优化，可以用后四个的峰值来确定位置
主要是后四个数是solid的

'''
#plt.show()


