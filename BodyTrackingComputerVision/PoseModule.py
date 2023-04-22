# Can use module in any project

import cv2
# google mediapipe
import mediapipe as mp
import time


class poseDetector():
    # pass in parameters for pose landmark detection and tracking
    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # Pose function() detection and tracking, true to enable detetion at all times, 
        # Option to track upper body only (33 landmarks (id: 0-32) vs 25 landmarks (id: 0-24))
        # self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
        self.pose = self.mpPose.Pose(self.mode, self.upBody, smooth_landmarks=self.smooth, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)


    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                # if it the landmarks are true (exist) draw the landmarks on the image
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            # Getting specific ID for each landmark
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)

                # getting pixel value from the ratio of where the lm is on the screen
                # type cast to integer as these are pixel values on the screen
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
            

def main():
    # Video capture with local camera
    cap  = cv2.VideoCapture(1)
    pTime = 0
    detector = poseDetector()

    while True:
        # Read camera
        success, img = cap.read()
        # Access findPose function in detector (poseDetector()) class
        img = detector.findPose(img) 
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            print(lmList) # could specify a specific landmark index vlaues we want to output ex. lmList[14]
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 14, (255, 255, 0), cv2.FILLED) # different visual feedback for specific point

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


        cv2.imshow("Image", img)

        cv2.waitKey(1)



# if we are running code by its self (just the file) it will execute the main function
# if we call another function it will not run the main code
if __name__ == "__main__":
    main()