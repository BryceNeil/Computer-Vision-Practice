import cv2
# google mediapipe
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
# Pose function() detection and tracking, true to enable detetion at all times, 
# Option to track upper body only (33 landmarks (id: 0-32) vs 25 landmarks (id: 0-24))
# 
pose = mpPose.Pose()


# Video capture with local camerax
cap  = cv2.VideoCapture(1)
pTime = 0

while True:
    # Read camera
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        # if it the landmarks are true (exist) draw the landmarks on the image
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Getting specific ID for each landmark\
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)

            # getting pixel value from the ratio of where the lm is on the screen
            # type cast to integer as these are pixel values on the screen
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    cv2.imshow("Image", img)

    cv2.waitKey(1)
