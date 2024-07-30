import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyfirmata

# Initialize video capture with USB webcam index (e.g., 1)
cap = cv2.VideoCapture(1)
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

# Set thresholds for each servo
minDegX, maxDegX = 75, 165  # Adjust these values for X servo
minDegY, maxDegY = 30, 150  # Adjust these values for Y servo
minDegZ, maxDegZ = 110, 130   # Reversed values for Z servo

servoX = 90  # Starting position
servoY = 90  # Starting position
servoZ = 90  # Starting position

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)  # Update findHands call to return both hands and image

    if hands:
        # Check for right hand
        if hands[0]['type'] == 'Right':
            hand_r = hands[0]
        elif len(hands) == 2 and hands[1]['type'] == 'Right':
            hand_r = hands[1]
        else:
            hand_r = None

        # Process right hand for x, y, and z-axis control
        if hand_r is not None:
            lmList_r = hand_r['lmList']  # List of 21 Landmark points
            
            # X-axis control (horizontal position)
            x_pos = lmList_r[9][0]  # Using middle finger MCP joint for stability
            servoX = np.interp(x_pos, [0, 1280], [minDegX, maxDegX])
            barX = np.interp(x_pos, [0, 1280], [25, 1255])
            
            # Y-axis control (vertical position)
            y_pos = lmList_r[9][1]
            servoY = np.interp(y_pos, [0, 720], [minDegY, maxDegY])
            barY = np.interp(y_pos, [0, 720], [25, 600])
            
            # Z-axis control (distance between thumb and index finger)
            length_z, info_z, img = detector.findDistance(lmList_r[4][0:2], lmList_r[8][0:2], img)
            servoZ = np.interp(length_z, [minHand, maxHand], [maxDegZ, minDegZ])
            barZ = np.interp(length_z, [minHand, maxHand], [600, 25])  # Inverse the bar for Z

            # Visual feedback
            posCircleX = int(np.interp(x_pos, [0, 1280], [25, 1255]))
            posCircleY = int(np.interp(y_pos, [0, 720], [25, 695]))
            cv2.circle(img, (posCircleX, posCircleY), 25, (0, 0, 255), cv2.FILLED)

            # Horizontal bar and servo degree for X servo at the bottom of the screen
            cv2.rectangle(img, (50, 650), (1230, 675), (255, 0, 0), 3)
            cv2.rectangle(img, (50, 650), (int(barX), 675), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoX)} deg', (50, 645), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            
            # Vertical bar and servo degree for Y servo on the left side of the screen
            cv2.rectangle(img, (25, 25), (50, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (25, int(barY)), (50, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoY)} deg', (55, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            
            # Vertical bar and servo degree for Z servo on the right side of the screen
            cv2.rectangle(img, (1205, 25), (1230, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (1205, int(barZ)), (1230, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoZ)} deg', (1050, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            # Servo control
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)
            servo_pinZ.write(servoZ)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
