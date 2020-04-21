import cv2
import imutils
import numpy as np
from PIL import Image


face_cascade = cv2.CascadeClassifier('./haarcascade_profileface.xml')
body_cascade = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
body1_cascade=cv2.CascadeClassifier('./haarcascade_upperbody.xml')

cap=cv2.VideoCapture('./NVR_ch3_main_20191228140000_20191228150000.asf')

while True:
    ret, image =cap.read()
    image = imutils.resize(image, width=min(800, image.shape[1]))
    image= cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bgsegm.createBackgroundSubtractorMOG();
    pick=body_cascade.detectMultiScale(gray,1.3,3)
    for (xA,yA,wA,hA) in pick:
        roi_gray=gray[yA:yA+int(hA/6), xA:xA+wA]
        roi_color=image[yA:yA+int(hA/6), xA:xA+wA]
        cv2.imwrite('img.jpg', roi_color)
        roi=Image.open("img.jpg")
        size=_w,_h=roi.size
        data=roi.load()
        color=[]
        for _x in range(_w):
            for _y in range(_h):
                color=data[_x,_y]
        mean = sum(color) / len(color)
        res = sum((i - mean) ** 2 for i in color) / len(color)
        print (res)
        #if res < 15: #edit this to suit ur conditon
            #cv2.rectangle(image, (xA,yA), (xA+wA,yA+hA), (225,0,0), 2)
        #if res > 70: #edit this to suit ur conditon
            #cv2.rectangle(image, (xA,yA), (xA+wA,yA+hA), (0,0,225), 2)
            
        cv2.rectangle(image, (xA,yA), (int((xA+wA)/1),int((yA+hA)/1)),(0,225,0),2)
    pick1=body1_cascade.detectMultiScale(gray,1.3,3)
    for (xA,yA,wA,hA) in pick1:
        roi_gray=gray[yA+int(hA/4):yA+int(hA/3), xA:xA+wA]
        roi_color=image[yA+int(hA/4):yA+int((hA/3)*2), xA:xA+wA]
        cv2.imwrite('img.jpg', roi_color)
        roi=Image.open("img.jpg")
        size=_w,_h=roi.size
        data=roi.load()
        color=[]
        for _x in range(_w):
            for _y in range(_h):
                color=data[_x,_y]
        mean = sum(color) / len(color)
        res = sum((i - mean) ** 2 for i in color) / len(color)
        print (res)
        #if res < 10: #edit this to suit ur conditon
            #cv2.rectangle(image, (xA,yA), (xA+wA,yA+hA), (225,0,0), 2)
        #if res > 50: #edit this to suit ur conditon
            #cv2.rectangle(image, (xA,yA), (xA+wA,yA+hA), (0,0,225), 2)
            
        cv2.rectangle(image, (xA,yA), (int((xA+wA)/1),int((yA+hA)/1)),(0,225,0),2)
            

    
    cv2.imshow('img',image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
