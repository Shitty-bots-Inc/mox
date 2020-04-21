import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')
cap= cv2.VideoCapture(0)
while True:
    ret, img=cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h) ,( 255,0,0), 2)
        roi_gray = gray[(y+h)/2:y+h, (x+w)/2:x+w]
        roi_color= img[y/2:y+h, x/2:x+w]
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh)in mouth:
            cv2.rectangle(roi_color, (mx,my), (mx+mw, my+mh), (0,225,0) , 2)
    cv2.imshow('img',img)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

     
