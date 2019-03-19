import cv2
import numpy as np

#cap = cv2.VideoCapture('ch02_20190312153219_edited.mp4')
#ret, img= cap.read()
img = cv2.imread('../pics/perspectivetransform/1552892153.95.jpg')
#print img.shape

image = img#[img.shape[0]/3:,img.shape[1]/3:img.shape[1]*2/3]
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print xy
        cv2.circle(image[img.shape[0]/2:,:img.shape[1]*2/3], (x, y), 1, (255, 105, 0), thickness = -1)
        cv2.putText(image[img.shape[0]/2:,:img.shape[1]*2/3], xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (105,105,0), thickness = 2)
        cv2.imshow("image", image[img.shape[0]/2:,:img.shape[1]*2/3])

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", image[img.shape[0]/2:,:img.shape[1]*2/3])

while(True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindow("image")
        break
        
cv2.waitKey(0)
cv2.destroyAllWindow()
