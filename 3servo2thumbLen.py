import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyfirmata

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize Arduino board
port = "COM5"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:5:s')  # pin 5 Arduino
servo_pinY = board.get_pin('d:6:s')  # pin 6 Arduino
servo_pinZ = board.get_pin('d:9:s')  # pin 9 Arduino

# Initialize variables
minHand, maxHand = 20, 220
minDeg, maxDeg = 0, 180
minBar, maxBar = 400, 150
servoX = 90  # Starting positio5
servoY = 90  # Starting position
servoZ = 90  # Starting position

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)  # Update findHands call to return both hands and image

    if hands:
        # Check for left hand
        if hands[0]['type'] == 'Left':
            hand_l = hands[0]
        elif len(hands) == 2 and hands[1]['type'] == 'Left':
            hand_l = hands[1]
        else:
            hand_l = None

        # Check for right hand
        if hands[0]['type'] == 'Right':
            hand_r = hands[0]
        elif len(hands) == 2 and hands[1]['type'] == 'Right':
            hand_r = hands[1]
        else:
            hand_r = None

        # Process left hand for x-axis control
        if hand_l is not None:
            lmList_l = hand_l['lmList']  # List of 21 Landmark points
            length_l, info_l, img = detector.findDistance(lmList_l[8][0:2], lmList_l[4][0:2], img)
            servoX = np.interp(length_l, [minHand, maxHand], [minDeg, maxDeg])
            barX = np.interp(length_l, [minHand, maxHand], [minBar, maxBar])

        # Process right hand for y-axis control
        if hand_r is not None:
            lmList_r = hand_r['lmList']  # List of 21 Landmark points
            length_r, info_r, img = detector.findDistance(lmList_r[8][0:2], lmList_r[4][0:2], img)
            servoY = np.interp(length_r, [minHand, maxHand], [maxDeg, minDeg])
            barY = np.interp(length_r, [minHand, maxHand], [minBar, maxBar])

        # Process distance between thumbs for z-axis control
        if hand_l is not None and hand_r is not None:
            length_z, info_z, img = detector.findDistance(lmList_l[4][0:2], lmList_r[4][0:2], img)
            length_z *= 2  # Double the threshold distance
            servoZ = np.interp(length_z, [minHand, maxHand * 2], [maxDeg, minDeg])  # Reverse the logic
            barZ = np.interp(length_z, [minHand, maxHand * 2], [minBar, maxBar])

        # Visual feedback
        if hand_l is not None:
            posCircleX = int(np.interp(length_l, [minHand, maxHand], [25, 1255]))
            cv2.circle(img, (posCircleX, 360), 25, (0, 0, 255), cv2.FILLED)

        if hand_r is not None:
            posCircleY = int(np.interp(length_r, [minHand, maxHand], [25, 695]))
            cv2.circle(img, (640, posCircleY), 25, (0, 0, 255), cv2.FILLED)

        # Bar display for left hand (x-axis)
        if hand_l is not None:
            cv2.rectangle(img, (1180, 150), (1215, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (1180, int(barX)), (1215, 400), (0, 255, 0), cv2.FILLED)

        # Bar display for right hand (y-axis)
        if hand_r is not None:
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(barY)), (85, 400), (0, 255, 0), cv2.FILLED)

        # Bar display for z-axis (distance between thumbs, horizontal bar at the top)
        if hand_l is not None and hand_r is not None:
            cv2.rectangle(img, (150, 50), (1130, 85), (255, 0, 0), 3)
            cv2.rectangle(img, (150, 50), (int(barZ), 85), (0, 255, 0), cv2.FILLED)

        # Servo control
        servo_pinX.write(servoX)
        servo_pinY.write(servoY)
        servo_pinZ.write(servoZ)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
