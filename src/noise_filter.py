#removing noise

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('1552929185.23.jpg')

dst1 = cv2.fastNlMeansDenoisingColored(img,None,100,100,7,21)
dst2 = cv2.fastNlMeansDenoisingColored(dst1,None,100,100,7,21)
#dst3 = cv2.fastNlMeansDenoisingColored(img,None,300,300,7,21)

cv2.imwrite('noise_filtered.jpg',dst2)
'''
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
'''