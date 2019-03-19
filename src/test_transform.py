import cv2
import numpy as np
import time
#import matplotlib.pyplot as plt
#cap = cv2.VideoCapture('ch02_20190312153219_edited.mp4')
#ret, img= cap.read()
img = cv2.imread('../pics/perspectivetransform/1552892190.9.jpg')

<<<<<<< HEAD
image = img#[img.shape[0]/3:,img.shape[1]/3:img.shape[1]*2/3]
cols,rows = image.shape[:2]
print(image.shape)
def get_perspective_mat():
 #159,242 tl 172 217 300,217 tr 300 217 305,285 br 300 285 161,321 bl 172 285
    #src_points = np.array([[223., 93.], [846., 10.], [302., 298.], [925., 215.]], dtype = "float32")
    #dst_points = np.array([[246., 10.], [846., 10.], [246., 252.], [846., 252.]], dtype = "float32")

    #src_points = np.array([[682., 650.], [1070., 651.], [1276., 494.], [1071., 495.]], dtype = "float32")
    #dst_points = np.array([[382., 800.], [382., 1000.], [1071., 1000.], [1071., 800.]], dtype = "float32")

    src_points = np.array([[884., 336.], [949., 269.], [1191., 478.], [1244., 380.]], dtype = "float32")
    dst_points = np.array([[844., 605.], [1000., 606.], [890., 1000.], [960., 985.]], dtype = "float32")
=======
image = cv2.imread('../pics/side_view_of_trucks/test.jpeg')
rows,cols = image.shape[:2]

def get_perspective_mat():
 
    src_points = np.array([[558., 314.], [654., 217.], [580., 474.], [673., 376.]], dtype = "float32")
    dst_points = np.array([[473., 240.], [663., 240.], [473., 370.], [663., 370.]], dtype = "float32")
 
>>>>>>> 00a8d5e1b2ef6b41080f1d1181bea9edb0b1378c
    M = cv2.getPerspectiveTransform(src_points, dst_points)
 
    return M

#pts1 = np.float32([[50,50],[200,50],[50,200]])
#pts2 = np.float32([[10,100],[200,50],[100,250]])
#M = cv2.getAffineTransform(pts1,pts2)
M = get_perspective_mat()
<<<<<<< HEAD
np.save('M', M)
res = cv2.warpPerspective(image, M, (rows, cols*2), cv2.INTER_LINEAR)
print(res.shape)
#res = cv2.warpAffine(img,M,(rows,cols))

#save_path = '../pics/perspectivetransform/' + str(time.time()) + '.jpg'
#cv2.imwrite(save_path, res)
cv2.imshow('frame',img)
cv2.imshow('res',res[500:1200,660:])
=======
res = cv2.warpPerspective(image, M, (rows, cols), cv2.INTER_LINEAR)
#res = cv2.warpAffine(img,M,(rows,cols))

save_path = '../pics/perspectivetransform/' + str(time.time()) + '.jpg'
cv2.imwrite(save_path, res)
#cv2.imshow('frame',res)
cv2.imshow('res',res)
>>>>>>> 00a8d5e1b2ef6b41080f1d1181bea9edb0b1378c

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()


