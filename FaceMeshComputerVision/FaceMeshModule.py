import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, staticMode = False, maxFaces = 4, minDetectionCon = 0.5, minTrackCon = 0.5):
       
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon


        self.mpDraw = mp.solutions.drawing_utils
        # use face mesh from library - 468 points total (0 - 467)
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius=2)


    def findFaceMesh(self, img, draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        # contains x and y output for each face -- look at making tuples inside list instead
        faces = []
        if self.results.multi_face_landmarks:
            

            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)


                # List for face variables, single face
                face = []
                # enumerate to go though each of numbers and ouput the id and correpsonding location of the point of given id
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    # print out id number of each of the points
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)


                    # print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(1)
    pTime = 0

    detector = FaceMeshDetector()


    while True:
        success, img = cap.read() 
        img, faces = detector.findFaceMesh(img)         # img = detector.findFaceMesh(img, False) if we do not want function to draw faceMesh

        if len(faces) != 0:
            # of faces that being detected in the frame
            print(len(faces))

            # print coordinates of face 0 thoughout time
            # print(faces[0])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()