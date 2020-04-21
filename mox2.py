import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./eye.xml')
mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
eye_cascade=cv2.CascadeClassifier('./eye.xml')
i=1
c=0
v=0
cap = cv2.VideoCapture(0)
while True:
    ret, img1 =cap.read()
    img= cv2.resize(img1, (900,600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in eyes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        v=v+1
        #cv2.rectangle(img, (x,y), (x+w*4,y+h*4), (0,255,0), 2)
        roi_gray=gray[y+h+1:y+3*h, x:x+w*3]
        roi_color=img[y+h+1:y+3*h, x:x+w*3]
        mouths = mouth_cascade.detectMultiScale(roi_gray,1.23,5)
        for (mx, my, mh, mw) in mouths:
             #cv2.rectangle(roi_color, (mx,my), ((mx+mw),(my+mh)), (0,255,0), 2)
             cv2.rectangle(img, (x,y), (x+w*4,y+h*4), (0,0,255), 2)
             i=0
        if i==0 :
            c=c+1
            i=1

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print('no mask:' + str(c))
        print('mask:' + str(v/2 -c))
        break

cap.release()
cv2.destroyAllWindows()
        
