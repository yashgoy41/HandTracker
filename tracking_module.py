import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity=1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        # this class only uses RGB images
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # information stored in results
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # each id has a corresponding landmark
                # we will use the x&y coordinates to get the landmark
                # numbers are given in decimals so we need to multiply by width and height
                # to get the pixel location
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self, img, position = 0, handNo=0, draw = True ):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                # coordinate of each point
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
        if draw and len(lmList) != 0:
            cv2.circle(img, (lmList[position][1], lmList[position][2]), 15, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    pTime = 0
    cTime = 0
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img,position=2)
        if len(lmList) != 0:
            print(lmList[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
