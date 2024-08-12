import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyfirmata
import time

# Initialize video capture with USB webcam index (e.g., 1)
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize Arduino board
port = "COM5"  # Updated COM port
board = pyfirmata.Arduino(port)

# Initialize servos for robot arm (X, Y, Z)
servo_pinX = board.get_pin('d:3:s')  # pin 3 Arduino
servo_pinY = board.get_pin('d:5:s')  # pin 5 Arduino
servo_pinZ = board.get_pin('d:6:s')  # pin 6 Arduino

# Initialize servos for movement (L and R)
servoL = board.get_pin('d:8:s')  # pin 8 Arduino
servoR = board.get_pin('d:9:s')  # pin 9 Arduino

# Initialize variables for robot arm
minHand, maxHand = 20, 220
minDegX, maxDegX = 60, 180
minDegY, maxDegY = 40, 140
minDegZ, maxDegZ = 100, 150
servoX = 120  # Starting position
servoY = 120  # Starting position
servoZ = 120  # Starting position

# Button positions
button_radius = 50
center_x, center_y = 640, 360  # Center of the screen
button_positions = {
    'forward': (center_x, center_y - 150),
    'reverse': (center_x, center_y + 150),
    'left': (center_x - 150, center_y),
    'right': (center_x + 150, center_y),
}

# Initialize servo positions
neutral_position = 90
servoL_position = neutral_position
servoR_position = neutral_position

# Time tracking for movement stop
last_movement_time = time.time()

def stop_servos():
    """Function to stop both movement servos."""
    servoL.write(neutral_position)
    servoR.write(neutral_position)

def move_servos(left_pos, right_pos):
    """Function to move both movement servos to specified positions."""
    servoL.write(left_pos)
    servoR.write(right_pos)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)  # Update findHands call to return both hands and image

    if hands:
        # Identify right and left hands
        hand_r = next((h for h in hands if h['type'] == 'Right'), None)
        hand_l = next((h for h in hands if h['type'] == 'Left'), None)

        # Draw the control buttons
        for button, pos in button_positions.items():
            cv2.circle(img, pos, button_radius, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, button, (pos[0] - 30, pos[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Process right hand for servoL and servoR control
        if hand_r is not None:
            lmList_r = hand_r['lmList']  # List of 21 Landmark points
            
            # Green dot on the top of the right hand
            wrist_x, wrist_y = lmList_r[0][0], lmList_r[0][1]  # Wrist position
            cv2.circle(img, (int(wrist_x), int(wrist_y - 30)), 10, (0, 255, 0), cv2.FILLED)  # Dot above wrist

            index_tip_x, index_tip_y = lmList_r[8][0], lmList_r[8][1]

            # Check if the index finger tip is inside any button
            for button, pos in button_positions.items():
                if np.hypot(index_tip_x - pos[0], index_tip_y - pos[1]) < button_radius:
                    if button == 'forward':
                        move_servos(30, 150)
                    elif button == 'reverse':
                        move_servos(150, 30)
                    elif button == 'left':
                        move_servos(0, 90)
                    elif button == 'right':
                        move_servos(90, 150)
                    last_movement_time = time.time()
                    break
            else:
                stop_servos()
                last_movement_time = time.time()

        # Stop servos if no valid hand position detected for 2 seconds
        if time.time() - last_movement_time >= 2:
            stop_servos()

        # Process left hand for X, Y, and Z control
        if hand_l is not None:
            lmList_l = hand_l['lmList']
            
            # X-axis control (horizontal position)
            x_pos = lmList_l[9][0]
            servoX = np.interp(x_pos, [0, 1280], [minDegX, maxDegX])
            barX = np.interp(x_pos, [0, 1280], [25, 1255])
            
            # Y-axis control (vertical position)
            y_pos = lmList_l[9][1]
            servoY = np.interp(y_pos, [0, 720], [minDegY, maxDegY])
            barY = np.interp(y_pos, [0, 720], [25, 600])
            
            # Z-axis control (distance between thumb and index finger)
            length_z, info_z, img = detector.findDistance(lmList_l[4][0:2], lmList_l[8][0:2], img)
            servoZ = np.interp(length_z, [minHand, maxHand], [maxDegZ, minDegZ])
            barZ = np.interp(length_z, [minHand, maxHand], [600, 25])  # Inverse the bar for Z

            # Visual feedback for left hand controls
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

            # Servo control for robotic arm
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)
            servo_pinZ.write(servoZ)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
