import cv2
import numpy as np

<<<<<<< HEAD
img = cv2.imread('../pics/2019-03-18-16:07:57.jpg')
#print img.shape
image = img[img.shape[0]//2:,0:]
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
=======
<<<<<<< HEAD
#cap = cv2.VideoCapture('ch02_20190312153219_edited.mp4')
#ret, img= cap.read()
img = cv2.imread('../pics/perspectivetransform/1552892153.95.jpg')
#print img.shape

image = img#[img.shape[0]/3:,img.shape[1]/3:img.shape[1]*2/3]
=======
img = cv2.imread('../pics/WechatIMG303.jpeg')
#print img.shape
image = img[img.shape[0]/2:,0:]
>>>>>>> 00a8d5e1b2ef6b41080f1d1181bea9edb0b1378c
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print xy
<<<<<<< HEAD
        cv2.circle(image[img.shape[0]/2:,:img.shape[1]*2/3], (x, y), 1, (255, 105, 0), thickness = -1)
        cv2.putText(image[img.shape[0]/2:,:img.shape[1]*2/3], xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (105,105,0), thickness = 2)
        cv2.imshow("image", image[img.shape[0]/2:,:img.shape[1]*2/3])

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", image[img.shape[0]/2:,:img.shape[1]*2/3])
=======
>>>>>>> 017b603460b5120cb3d400e13af2e5f135b6bd39
        cv2.circle(image, (x, y), 1, (255, 105, 0), thickness = -1)
        cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (105,105,0), thickness = 2)
        cv2.imshow("image", image)

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", image)
>>>>>>> 00a8d5e1b2ef6b41080f1d1181bea9edb0b1378c

while(True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindow("image")
        break
        
cv2.waitKey(0)
cv2.destroyAllWindow()
