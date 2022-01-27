import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

mpHands = mp.solutions.hands
# this class only uses RGB images
hands = mpHands.Hands()

pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # information stored in results
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    mpDraw = mp.solutions.drawing_utils

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # each id has a corresponding landmark
            # we will use the x&y coordinates to get the landmark
            # numbers are given in decimals so we need to multiply by width and height
            # to get the pixel location
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                # coordinate of each point
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # calculating frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
