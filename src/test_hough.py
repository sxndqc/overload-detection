# import the necessary packages
import numpy as np
import argparse
import cv2
import time

#cap = cv2.VideoCapture(0) # Set Capture Device, in case of a USB Webcam try 1, or give -1 to get a list of available devices

#Set Width and Height 
# cap.set(3,1280)
# cap.set(4,720)

# The above step is to set the Resolution of the Video. The default is 640x480.
# This example works with a Resolution of 640x480.


# Capture frame-by-frame
frame = cv2.imread('../pics/truck14.jpg')
size = frame.shape
# load the image, clone it for output, and then convert it to grayscale
#frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)	
#print frame.shape	
output = frame.copy()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#print gray
# apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
#gray = cv2.GaussianBlur(gray,(5,5),0)
#gray = cv2.medianBlur(gray,5)

# Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
#gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         #cv2.THRESH_BINARY,11,3.5)

#kernel = np.ones((2.6,2.7),np.uint8)
#gray = cv2.erode(gray,kernel,iterations = 1)

#gray = cv2.dilate(gray,kernel,iterations = 1)
# gray = dilation
gray = cv2.Sobel(gray,-1, 0, 1)
img = gray[size[0]/2:,0:]
# get the size of the final image
# img_size = gray.shape
# print img_size

# detect circles in the image
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=100, param2=50, minRadius=5, maxRadius=50)#param2=50 is better for most cases
# print circles

# ensure at least some circles were found
if circles is not None:
# convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

# loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle in the image
        # corresponding to the center of the circle
        cv2.circle(output, (x, y+size[0]/2), r, (0, 255, 0), 4)
        #cv2.rectangle(output, (x - r-5, y+size[0]/2-r - 5), (x +r+ 5, y +r+size[0]/2+ 5), (0, 128, 255), 2)
        #time.sleep(0.5)
        

# Display the resulting frame
    print len(circles)
    cv2.imshow('gray',gray)
    cv2.imshow('frame',output)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()