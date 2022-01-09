import time
import mediapipe as mp
import cv2

print("setup complete , good to goo!")

cap = cv2.VideoCapture("vids/pose-fight.mp4")
ptime =0
mpFaceDet = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDet = mpFaceDet.FaceDetection(0.75)


while 1:
    s,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = FaceDet.process(imgRGB)
    if results.detections:
        for id,det in enumerate(results.detections):
            h,w,c = img.shape
            boxcoord = det.location_data.relative_bounding_box
            box = int(boxcoord.xmin*w) , int(boxcoord.ymin*h) , int(boxcoord.width*w) , int(boxcoord.height*h)
            cv2.rectangle(img,box,(255, 0, 0),2)
            cv2.putText(img, f'{round(100*det.score[0],3)}', (box[0], box[1]-20), 2, cv2.FONT_HERSHEY_PLAIN, (255, 0, 0), 2)
            #print(det)
            #mpDraw.draw_detection(img,det)  #this statemtn is used to draw a square using a builtin method
            #print(id)
            #print("The machine is ",round(100*det.score[0],4),"% sure that the current frame has a face")

    ctime = time.time()
    fps=0
    if ctime-ptime:
        fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,"FPS "+str(int(fps)),(50,50),5,cv2.FONT_HERSHEY_DUPLEX,(0,255,0),3)
    cv2.imshow("face detection",img)
    cv2.waitKey(20)