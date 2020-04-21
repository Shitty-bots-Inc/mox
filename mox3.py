import cv2
import numpy as np
from PIL import Image
from collections import Counter

mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
eye_cascade=cv2.CascadeClassifier('./eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    #img1=img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in eyes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_gray=gray[y+h+1:y+3*h, x:x+w*3]
        roi_color=img[y+h+1:y+3*h, x:x+w*3]
        cv2.imwrite('im.jpg', roi_color)
        roi=Image.open("im.jpg")
        size=_w,_h=roi.size
        data=roi.load()
        color=[]
        for _x in range(_w):
            for _y in range(_h):
                color=data[_x,_y]
        #print (color)
        mean = sum(color) / len(color) 
        res = sum((i - mean) ** 2 for i in color) / len(color)
        if res < 15: #edit this to suit ur conditon
            cv2.rectangle(img, (x,y), (x+w*4,y+h*4), (225,0,0), 2)
        if res > 100: #edit this to suit ur conditon
            cv2.rectangle(img, (x,y), (x+w*4,y+h*4), (0,0,225), 2)
            
                
       
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
        

