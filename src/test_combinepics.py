import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt
M = np.load('M.npy')
print(M)
previous = None
cap = cv2.VideoCapture('ch02_20190312153219_edited.mp4')
count = 0
width = 780 - 750
length = 1080 - 500

while True:
    ret, image= cap.read()
    #print(image.shape)
    if not ret:
        break
            
            #process the video or images
    count += 1
    #cols ,rows = image.shape[:2]
    #cv2.imshow('source',image)
    cols ,rows = image.shape[:2]
    newImg = Image.new("RGB",(width*count,length))
    #res = cv2.warpPerspective(image, M, (rows, cols*2), cv2.INTER_LINEAR)
    #print(res.shape)
    res = image[500:1080,750:780]
    #print(res.shape)
    if previous:
        newImg.paste(previous,(0,0))
    
    #size = tuple([r*4,0,r1*4,cols])
    #print(r,r1,cols)
    newImg.paste(Image.fromarray(res),(width*(count-1),0,width*count,length))
    #print(res)
    '''
    plt.imshow(newImg)
    plt.pause(10)
    plt.axis('off')
    plt.close() 
    '''
    cv2.imshow('res',res)
    previous = newImg
    cv2.waitKey(1)
#last = Image.fromarray(res)
#last.save("last.jpg")
dst = cv2.fastNlMeansDenoisingColored(newImg,None,100,100,7,21)
dst.save(str(time.time())+".jpg") 
cap.release()
cv2.destroyAllWindows() 
    