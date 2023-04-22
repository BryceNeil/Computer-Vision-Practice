import cv2
import mediapipe as mp
import time


class FaceDetector():

    # initializations
    def __init__(self, minDetectionCon = 0.5):
        # self meaning it is now an instance varaible (not generic or global variable)

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils

        # the number is by default 0.5, minimum detection confidence higher value can help remove false positives
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)


    def findFaces(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)

        bboxs = []

        if self.results.detections:
            for id, detection  in enumerate(self.results.detections):
              
                # The Coordinates, xmin, ymin, width, and height of the bounding box on the face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape

                # bounding box - allow for shorter call to get values
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                
                # detection information 
                bboxs.append([id, bbox, detection.score])

                # Function to draw round face
                if draw:
                    self.fancyDraw(img, bbox)
               
                    # detection confidence score appended to top left of face box
                    cv2.putText(img, f'Confidence Rate: {int(detection.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0 , 0), 2)

        # return bounding bboxs and the image that hs the detections on it
        return img, bboxs
    
 
    def fancyDraw(self, img, bbox, l = 40, t = 5, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        # drawing by ourselves instead of using default function to draw bounding box
        cv2.rectangle(img, bbox, (255, 0, 0), rt)

        # Top Left x,y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 0), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 0), t)

        # Top Right x,y
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 0), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 0), t)

        # Bottom Left x,y
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 0), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 0), t)

        # Bottom Right x,y
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 0), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 0), t)

        return img

def main():
    cap = cv2.VideoCapture(1)
    pTime = 0
    detector = FaceDetector()
 
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img) # To not draw: img, bboxs = detector.findFaces(img, False)


        # prints face #, bounding location and size, and confidence the object is a face
        print(bboxs)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0 , 0), 2)
        
        cv2.imshow("Image", img)

        cv2.waitKey(1) # number in here can control the framerate



if __name__ == "__main__":
    main()