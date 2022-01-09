import time
import mediapipe as mp
import cv2
import facedetmodule as fdm

cap = cv2.VideoCapture(0)
ptime =0
FaceDetn = fdm.DetectFace()
while 1:
    s,img = cap.read()
    img,boxdim = FaceDetn.DetectFaces(img)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f'FPS:{int(fps)}',(50,50),3,cv2.FONT_HERSHEY_PLAIN,(0,0,255),2)
    cv2.imshow("Face Detector",img)
    cv2.waitKey(1)