import m3u8
import requests
import time
import cv2
import numpy as np

def getM3u8(url):
    m3u8_obj = m3u8.load(url)  # this could also be an absolute filename

    ts_url_list = []

    base_uri = m3u8_obj.base_uri

    ts_list = m3u8_obj.files

    for _ts in ts_list:

        ts_url = base_uri + _ts

        ts_url_list.append(ts_url)

    return ts_list

def capture(ts_url_list):
    print(ts_url_list)
    first_frame = 0
    previous = None
    path = '../pics/test_pass_detector/'
    for ts in ts_url_list:
        cap = cv2.VideoCapture(ts)
        ret, image= cap.read()
        if not ret:
            return finished
        
        #process the video or images
        if first_frame != 0:
            coeff = frames_difference_check(image, previous)
            
            if coeff < 0.9:
                timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                cv2.imwrite(path+str(timestamp)+'_c'+'.jpg', image)
                cv2.imwrite(path+str(timestamp)+'_p'+'.jpg', previous)  
                print('pass found!')
        else:
            first_frame += 1

        previous = image              
        # When everything done, release the capture
        #cap.release()
   
def frames_difference_check(current,previous):
    
    img0= previous.reshape(previous.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
    img1= current.reshape(current.size, order='C')

    print("Correlation coefficient of image 0 and image 1: %f\n" % np.corrcoef(img0, img1)[0, 1])
    return np.corrcoef(img0, img1)[0, 1]

if __name__ == "__main__":
    #TODO: to request accessToken and fresh it
    url = "http://hls.open.ys7.com/openlive/ddb43495ceb64e8fa3c136c5737915cc.m3u8"
    while True:
        ts_list = getM3u8(url)
        capture(ts_list)
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows() 