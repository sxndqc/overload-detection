import cv2
import numpy as np
#import matplotlib.pyplot as plt

img = cv2.imread('../pics/side_view_of_trucks/test.jpeg')
rows,cols = img.shape[:2]
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)

res = cv2.warpAffine(img,M,(rows,cols))
cv2.imshow('res',res)
cv2.imshow('frame',img)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()


