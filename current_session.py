# coding: utf-8
get_ipython().magic('cd cvtest/')
import cv2
a = cv2.imread("ttest.jpg")
a.shape
aa = a[500:580, 580:700]
cv2.imwrite('aa.jpg',aa)
get_ipython().magic('ls ')
get_ipython().magic('cp aa.jpg /mnt/d/')
aa = a[540:620, 680:800]
cv2.imwrite('aa.jpg',aa)
get_ipython().magic('cp aa.jpg /mnt/d/')
aa = a[560:600, 680:740]
cv2.imwrite('aa.jpg',aa)
get_ipython().magic('cp aa.jpg /mnt/d/')
get_ipython().magic('ls ')
get_ipython().magic('cd cv')
get_ipython().magic('cd ..')
get_ipython().magic('cd cv')
get_ipython().magic('ls ')
get_ipython().magic('cp /mnt/d/aa.jpg ./')
get_ipython().magic('ls ')
get_ipython().magic('cp /mnt/d/aa.jpg ./')
get_ipython().magic('cp /mnt/d/aa.jpg ./')
get_ipython().magic('ls ')
aa
a.shape
aa.shape
ar = cv2.getRotationMatrix((20,30),1)
ar = cv2.getRotationMatrix2D((20,30),1)
ar = cv2.getRotationMatrix2D((20,30),30,1)
aar = cv2.warpAffine(aa,ar,(20,30))
aar.imwrite("aar.jpg")
cv2.imwrite("aar.jpg",aar)
aar.shape
ar = cv2.getRotationMatrix2D((30,20),30,1)
aar = cv2.warpAffine(aa,ar,(30,20))
cv2.imwrite("aar.jpg",aar)
aar = cv2.warpAffine(aa,ar,(40,60))
cv2.imwrite("aar.jpg",aar)
aar = cv2.warpAffine(aa,ar,(60,40))
cv2.imwrite("aar.jpg",aar)
aaa = aar[10:30,20:50]
cv2.imwrite("aaa.jpg",aaa)
get_ipython().magic('cp ../cvtest/ocr.py ./')
warped = cv2.imread("aaa.jpg")
warped =  cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
output = image
output = aaa
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
thresh
thresh = thresh[1]
cv2.imwrite("thresh.jpg",thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel)
cv2.imwrite("thresh.jpg",thresh)
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
import imutils
cnts = imutils.grab_contours(cnts)
digitCnts = []
cnts
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    if w>=8 and (h>=25 and h<=38):
        digitCnts.append(c)
        
digitCnts
cc = [cv2.boundingRect(c) for c in cnts]
cc
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel)
cv2.imwrite("thresh.png",thresh)
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imwrite("thresh.bmp",thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,2))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imwrite("thresh.jpeg",thresh)
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    if w>=8 and (h>=25 and h<=38):
        digitCnts.append(c)
        
        
digitCnts
cc = [cv2.boundingRect(c) for c in cnts]
cc
thresh = cv2.imread("thresh.bmp")
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,2))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,4))
thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)
thresh3 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel3)
thresh4 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel4)
cnts = cv2.findContours(thresh2.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = cv2.findContours(thresh2.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts3 = cv2.findContours(thresh3.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts4 = cv2.findContours(thresh4.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = imutils.grab_contours(cnts2)
cnts3 = imutils.grab_contours(cnts3)
cnts4 = imutils.grab_contours(cnts4)
cc2 = [cv2.boundingRect(c) for c in cnts2]
cc3 = [cv2.boundingRect(c) for c in cnts3]
cc4 = [cv2.boundingRect(c) for c in cnts4]
cc2
cc3
cc4
from imutils import contours
digitCnts3 = contours.sort_contours(cnts3, method = "left-to-right")[0]
output3 = output
for (x,y,w,h) in cnts3:
    cv2.rectangle(output3, (x,y), (x+w,y+h), (0,255,0), 1)
    
for x,y,w,h in cnts3:
    cv2.rectangle(output3, (x,y), (x+w,y+h), (0,255,0), 1)
    
for i,(x,y,w,h) in enumerate(cnts3):
    cv2.rectangle(output3, (x,y), (x+w,y+h), (0,255,0), 1)
    
    
cnts3
for i,(x,y,w,h) in enumerate(cc3):
    cv2.rectangle(output3, (x,y), (x+w,y+h), (0,255,0), 1)
    
    
cv2.imwrite(output3)
cv2.imwrite("output3.jpg",output3)
blurred = cv2.GaussianBlur(warped, (3,3),0)
function w(filename):
def w(name):
    cv2.imwrite(name+'.jpg', locals()[name])
    
w('blurred')
blurred
w('blurred')
def w(name):
    cv2.imwrite(name+'.jpg', vars()[name])
    
    
w('blurred')
vars()['blurred']
cv2.imwrite('blurred.jpg',blurred)
def w(name):
    cv2.imwrite(name+'.jpg', globals()[name])
    
    
w('blurred')
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(output, contours, 0 ,(0,255,0),3)
b = cv2.drawContours(output, contours, 0 ,(0,255,0),3)
cnts
b = cv2.drawContours(output, cnts, 0 ,(0,255,0),3)
b = cv2.drawContours(output, cnts[0], 0 ,(0,255,0),3)
b
w('b'
)
len(cnts[0])
len(cnts[1])
len(cnts)
b = cv2.drawContours(output, cnts[0], 0 ,(0,255,0),2)
w('b')
b = cv2.drawContours(output, cnts[0], 1 ,(0,255,0),3)
w('b')
b = cv2.drawContours(output, cnts[0], -1,(0,255,0),-1)
w('b')
b = cv2.drawContours(output, cnts[0], 0,(0,255,0),-1)
w('b')
b = cv2.drawContours(output, cnts[0], 1,(0,255,0),-1)
w('b')
b = cv2.drawContours(output, cnts[0], 1,(0,255,0),1)
w('b')
w('output')
output = aaa
b = cv2.drawContours(output.copy(), cnts[0], 1,(0,255,0),1)
w('b')
w('aaa')
w('warped')
aaa = warped.copy()
output = aaa.copy()
b = cv2.drawContours(output.copy(), cnts[0], 1,(0,255,0),1)
w('b')
b = cv2.drawContours(output.copy(), cnts[0], 1,(0,255,0),2)
w('b'）
w('b')
get_ipython().magic('history')
get_ipython().magic('history > history.txt')
get_ipython().magic('history > ./history.txt')
get_ipython().magic('ls ')
touch history.txt
get_ipython().magic('save current_session ~0/')
