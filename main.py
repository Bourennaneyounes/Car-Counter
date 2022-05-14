import cv2
import numpy as np
from time import sleep

min_hieght_react=80 
min_width_react=80 
offset=6 

count_line_position=550 

delay= 10 

detect = []
counter= 0

	
def center_hendler(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
algo = cv2.createBackgroundSubtractorMOG2()

while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    counterShape,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255,127,0), 3) 
    for(i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counterShape = (w >= min_hieght_react) and (h >= min_width_react)
        if not validate_counterShape:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        center = center_hendler(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0,127,255), 3)  
                detect.remove((x,y))
                print("car is detectted : "+str(counter))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    # cv2.imshow("detecttar",dilatada)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
