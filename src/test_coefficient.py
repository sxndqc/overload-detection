
import cv2
import numpy as np
import time

def capture():
    first_frame = 0
    previous = None
    path = '../pics/test_pass_detector/'
    #for ts in ts_url_list:
    cap = cv2.VideoCapture('../pics/ts/test.mp4')
    while True:
        ret, image= cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return 'finished'
        
        image = image[0:,image.shape[1]//2:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #process the video or images
        if first_frame != 0:
            coeff = frames_difference_check(image, previous)
            
            if coeff < 0.8:
                timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                cv2.imwrite(path+str(coeff)+'_c'+'.jpg', image)
                cv2.imwrite(path+str(coeff)+'_p'+'.jpg', previous)  
                print('pass found!')
            
        else:
            first_frame += 1

        previous = image   
        #time.sleep(1)           
        # When everything done, release the capture
        #cap.release()
   
def frames_difference_check(current,previous):
    
    img0= previous.reshape(previous.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
    img1= current.reshape(current.size, order='C')

    print("Correlation coefficient of image 0 and image 1: %f\n" % np.corrcoef(img0, img1)[0, 1])
    return np.corrcoef(img0, img1)[0, 1]

if __name__ == "__main__":
    #TODO: to request accessToken and fresh it
    #url = "http://hls.open.ys7.com/openlive/ddb43495ceb64e8fa3c136c5737915cc.m3u8"
    
    capture()
    #time.sleep(1)

     