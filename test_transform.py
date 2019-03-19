import cv2
import numpy as np
import time
#import matplotlib.pyplot as plt

image = cv2.imread('test.jpg')
rows,cols = image.shape[:2]

def get_perspective_mat():
 
    src_points = np.array([[681., 547.], [676., 565.], [736., 580.], [731., 599.]], dtype = "float32")
    dst_points = np.array([[681., 540], [681., 560.], [736.,540.], [736., 560.]], dtype = "float32")
 
    M = cv2.getPerspectiveTransform(src_points, dst_points)
 
    return M

#pts1 = np.float32([[50,50],[200,50],[50,200]])
#pts2 = np.float32([[10,100],[200,50],[100,250]])
#M = cv2.getAffineTransform(pts1,pts2)
M = get_perspective_mat()
np.save("M.dat",M)
res = cv2.warpPerspective(image, M, (rows, cols), cv2.INTER_LINEAR)
#res = cv2.warpAffine(img,M,(rows,cols))

#save_path = '../pics/perspectivetransform/' + str(time.time()) + '.jpg'
cv2.imwrite("testback.jpg", res)
#cv2.imshow('frame',res)
#cv2.imshow('res',res)
im = cv2.resize(res[540:560, 696:736],(600,200))
cv2.imwrite("testblack.jpg", im)
"""
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()
"""
cv2.imwrite("test1.jpg", im[50:190,150:250])
cv2.imwrite("test2.jpg", im[50:190,240:340])
cv2.imwrite("test3.jpg", im[50:190,330:430])
cv2.imwrite("test4.jpg", im[50:190,420:520])
