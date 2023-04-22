import time
# The module we import must be in the same folder as the project
import PoseModule as pm 
import cv2

# THIS IS THE TESTING CODE - FROM MAIN() in MODULE
# Video capture with local camera
cap  = cv2.VideoCapture(1)
pTime = 0
detector = pm.poseDetector()

while True:
    # Read camera
    success, img = cap.read()
    # Access findPose function in detector (poseDetector()) class
    img = detector.findPose(img) 
    lmList = detector.findPosition(img, draw=False)

    # if there are elements lmList then
    if len(lmList) != 0:
        print(lmList) # could specify a specific landmark index vlaues we want to output ex. lmList[14]
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 14, (255, 255, 0), cv2.FILLED) # different visual feedback for specific point

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    cv2.imshow("Image", img)

    cv2.waitKey(1)