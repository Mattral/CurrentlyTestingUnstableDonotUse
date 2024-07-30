import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize video capture with USB webcam index (e.g., 1)
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize variables
minHand, maxHand = 20, 220

# Set thresholds for each axis
minDegX, maxDegX = 75, 165  # Adjust these values for X axis
minDegY, maxDegY = 30, 150  # Adjust these values for Y axis
minDegZ, maxDegZ = 80, 120  # Reversed values for Z axis

servoX = 90  # Starting position
servoY = 90  # Starting position
servoZ = 90  # Starting position

def control_motor(fingers):
    if fingers == 1:
        return "Forward"
    elif fingers == 2:
        return "Reverse"
    elif fingers == 4:
        return "Left Turn"
    elif fingers == 5:
        return "Right Turn"
    else:
        return "Stop"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)  # Update findHands call to return both hands and image

    if hands:
        hand_r, hand_l = None, None

        # Check for right and left hands
        for hand in hands:
            if hand['type'] == 'Right':
                hand_r = hand
            elif hand['type'] == 'Left':
                hand_l = hand

        # Process right hand for servo control
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

            # Visual feedback for right hand
            posCircleX = int(np.interp(x_pos, [0, 1280], [25, 1255]))
            posCircleY = int(np.interp(y_pos, [0, 720], [25, 695]))
            cv2.circle(img, (posCircleX, posCircleY), 25, (0, 0, 255), cv2.FILLED)

            # Horizontal bar for X axis at the bottom of the screen
            cv2.rectangle(img, (50, 650), (1230, 675), (255, 0, 0), 3)
            cv2.rectangle(img, (50, 650), (int(barX), 675), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoX)} deg', (50, 645), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            # Vertical bar for Y axis on the left side of the screen
            cv2.rectangle(img, (25, 25), (50, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (25, int(barY)), (50, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoY)} deg', (55, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            # Vertical bar for Z axis on the right side of the screen
            cv2.rectangle(img, (1205, 25), (1230, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (1205, int(barZ)), (1230, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoZ)} deg', (1050, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        # Process left hand for motor control
        if hand_l is not None:
            lmList_l = hand_l['lmList']
            fingers_l = detector.fingersUp(hand_l)
            motor_action = control_motor(fingers_l.count(1))

            # Display motor action on the screen
            cv2.putText(img, motor_action, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
