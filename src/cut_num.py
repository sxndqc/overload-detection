import os
import time
import cv2
import numpy as np

def get_perspective_mat():
  
    src_points = np.array([[688., 551.], [684., 570.], [749., 581.], [746., 601.]], dtype = "float32")
    dst_points = np.array([[681., 540.], [681., 560.], [736.,540.], [736., 560.]], dtype = "float32")
 
    M = cv2.getPerspectiveTransform(src_points, dst_points)
 
    return M
    
def num_segmentation(img):
    

if __name__ == "__main__":

    M = get_perspective_mat()
    np.save("M.dat",M)    

    url = "http://hls.open.ys7.com/openlive/77bd93fe20054c8893f9f0b309f83e31.hd.m3u8"

    while True:
    
        timestamp = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

        os.system('ffmpeg -probesize 32768 -i '+ url + ' -y -t 0.001 -f image2 -ss 1 -r 1 ../pics/'+ timestamp +'.jpg')
        
        path1 = "../pics/num_whole/"
        path2 = "../pics/num_sep/"
     
        image = cv2.imread('../pics/'+ timestamp +'.jpg')
        rows,cols = image.shape[:2]
        
        res = cv2.warpPerspective(image, M, (rows, cols), cv2.INTER_LINEAR)

        im = cv2.resize(res[540:560, 696:736],(600,200))
        
        cv2.imwrite(path1 + timestamp + '.jpg', im)
        
        cv2.imwrite(path2 + timestamp + "1.jpg", im[50:190, 60:160])
        cv2.imwrite(path2 + timestamp + "2.jpg", im[50:190,150:250])
        cv2.imwrite(path2 + timestamp + "3.jpg", im[50:190,240:340])
        cv2.imwrite(path2 + timestamp + "4.jpg", im[50:190,330:430])
        cv2.imwrite(path2 + timestamp + "5.jpg", im[50:190,420:520])
        
        time.sleep(4)
        
        
       

    
