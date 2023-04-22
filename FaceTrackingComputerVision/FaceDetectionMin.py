import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

# the number is by default 0.5, minimum detection confidence higher value can help remove false positives
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)


    if results.detections:
        for id, detection  in enumerate(results.detections):
            # draws bounding box and landmarks on face
            # mpDraw.draw_detection(img, detection)
            
            # print(id, detection)
            # Certainty that the detected object is a face
            # print(detection.score)

            # The Coordinates, xmin, ymin, width, and height of the bounding box on the face
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            # bounding box - allow for shorter call to get values
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            # drawing by ourselves instead of using default function to draw bounding box
            cv2.rectangle(img, bbox, (255, 0, 0), 2)
            # detection confidence score appended to top left of face box
            cv2.putText(img, f'FPS: {int(detection.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0 , 0), 2)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0 , 0), 2)
    
    cv2.imshow("Image", img)

    cv2.waitKey(1) # number in here can control the framerate

 