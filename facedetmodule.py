import time
import mediapipe as mp
import cv2
print('setup complete good to go!!!')

class DetectFace():

    def __init__(self, min_detection_confidence=0.75, model_selection=0):
        self.min_detection_confidence=min_detection_confidence
        self.model_selection = model_selection
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face = self.mpFace.FaceDetection(self.min_detection_confidence,self.model_selection)

    def DetectFaces(self,img,draw=True):
        ret = []
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        res = self.face.process(imgRGB)
        if res.detections:
            for id,det in enumerate(res.detections):
                h,w,c = img.shape
                boxx = det.location_data.relative_bounding_box
                box = int(boxx.xmin*w) , int(boxx.ymin*h) , int(boxx.width*w) , int(boxx.height*h)
                img = self.cherryontop(img,box)
                ret.append([id,box,det.score])
                cv2.putText(img,f'ACC:{int(100*det.score[0])}',(box[0],box[1]-20),5,cv2.FONT_HERSHEY_PLAIN,(0,255,0),2)

        return img,ret

    def cherryontop(self,img,bbox,l=30,t=5):
        x,y,w,h = bbox
        cv2.rectangle(img, bbox, (0, 255, 0), 1)
        #the next two lines is to draw a top left corner
        cv2.line(img,(x,y),(x+l,y),(0, 255, 0),t)
        cv2.line(img,(x,y),(x,y+l),(0, 255, 0),t)
        #to draw top right corner
        cv2.line(img, (x+w, y), (x+w-l, y), (0, 255, 0), t)
        cv2.line(img, (x+w, y), (x+w, y + l), (0, 255, 0), t)
        #to draw bottom left corner
        cv2.line(img, (x, y+h), (x + l, y+h), (0, 255, 0), t)
        cv2.line(img, (x, y+h), (x, y+h-l), (0, 255, 0), t)
        #to draw bottom right corner
        cv2.line(img, (x+w, y+h), (x+w - l, y+h), (0, 255, 0), t)
        cv2.line(img, (x+w, y+h), (x+w, y - l+h), (0, 255, 0), t)
        return img


def main():

    cap = cv2.VideoCapture(0)
    ptime =0
    Face = DetectFace()
    while 1:
        s,img = cap.read()
        img,boxdim = Face.DetectFaces(img)
        ctime = time.time()
        fps=0
        if ctime-ptime:
            fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(50,70),3,cv2.FONT_HERSHEY_DUPLEX ,(255,0,0),2)
        cv2.imshow("FaceDetection",img)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
