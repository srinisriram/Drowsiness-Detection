import cv2
import os
cap= cv2.VideoCapture(0)
i=0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i !=1000:
        cv2.imwrite('Fwame'+str(i)+'.jpg',frame)
        i+=1
        
cap.release()
cv2.destroyAllWindows()
