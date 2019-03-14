import cv2
import numpy as np
import time
#import matplotlib.pyplot as plt

image = cv2.imread('../pics/side_view_of_trucks/test.jpeg')
rows,cols = image.shape[:2]

def get_perspective_mat():
 
    src_points = np.array([[558., 314.], [654., 217.], [580., 474.], [673., 376.]], dtype = "float32")
    dst_points = np.array([[473., 240.], [663., 240.], [473., 370.], [663., 370.]], dtype = "float32")
 
    M = cv2.getPerspectiveTransform(src_points, dst_points)
 
    return M

#pts1 = np.float32([[50,50],[200,50],[50,200]])
#pts2 = np.float32([[10,100],[200,50],[100,250]])
#M = cv2.getAffineTransform(pts1,pts2)
M = get_perspective_mat()
res = cv2.warpPerspective(image, M, (rows, cols), cv2.INTER_LINEAR)
#res = cv2.warpAffine(img,M,(rows,cols))

save_path = '../pics/perspectivetransform/' + str(time.time()) + '.jpg'
cv2.imwrite(save_path, res)
#cv2.imshow('frame',res)
cv2.imshow('res',res)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()


