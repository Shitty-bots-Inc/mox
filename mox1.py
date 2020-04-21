import cv2
import numpy as np
#import face_recognition

#image = face_recognition.load_image_file("my_picture.jpg")
face_cascade = cv2.CascadeClassifier('./haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
eye_cascade=cv2.CascadeClassifier('./eye.xml')
check = cv2.imread('./check.jpeg')
font = cv2.FONT_HERSHEY_SIMPLEX 

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    #img.set(3,640)
    #img.set(4,480)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_recognition.face_locations(gray)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5
    eyes=eye_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in eyes :
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        roi_gray=gray[y+h+1:y+3*h, x:x+w*3]
        roi_color=img[y+h+1:y+3*h, x:x+w*3]
        blur = cv2.GaussianBlur(roi_gray,(5,5),0)
        median = cv2.medianBlur(blur,9)
        #dimensions = median.shape
        #big = cv2.resize(check,dimensions )
        #diff=cv2.subtract(big,check)
        #b,g,r = cv2.split(diff)
        #if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            #cv2.puttext(img, 'mask', (x, y+h+3),font , 1, (0,225,0), 3)
        mouths = mouth_cascade.detectMultiScale(roi_gray,1.23,5)
        for (mx, my, mh, mw) in mouths:
             cv2.rectangle(roi_color, (mx,my), (mx+mw,my+mh), (0,0,255), 2)           

        roi_color=img[y+h+1:y+3*h, x:x+w]
        cv2.rectangle(img, (x,y+h), (x+w*3,y+3*h), (0,255,0), 2)
                
        #roi_gray = gray[y+int(h/2):y+h, x:x+w]
        #roi_color = img[y+int(h/2):y+h, x:x+w]
        #mouths = mouth_cascade.detectMultiScale(roi_gray, 1.23, 5)
        #    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
        
