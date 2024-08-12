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
port = "COM6"  # Updated COM port
board = pyfirmata.Arduino(port)

# Initialize servos for robot arm (X, Y, Z)
servo_pinX = board.get_pin('d:3:s')  # pin 3 Arduino
servo_pinY = board.get_pin('d:5:s')  # pin 5 Arduino
servo_pinZ = board.get_pin('d:6:s')  # pin 6 Arduino

# Initialize servos for movement (L and R)
servoL = board.get_pin('d:9:s')  # pin 9 Arduino
servoR = board.get_pin('d:10:s')  # pin 10 Arduino

# Initialize variables for robot arm
minHand, maxHand = 20, 220
minDegX, maxDegX = 60, 180
minDegY, maxDegY = 40, 140
minDegZ, maxDegZ = 100, 150
servoX = 120  # Starting position
servoY = 120  # Starting position
servoZ = 120  # Starting positions

# Define the dimensions and positions of the boxes
center_x, center_y = 640, 360  # Center of the screen
half_screen_width = 640  # Half of the screen width
half_screen_height = 360  # Half of the screen height

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

        # Draw the control boxes
        cv2.rectangle(img, (0, 0), (half_screen_width, 720), (255, 0, 0), 2)  # Left box for L and R
        cv2.rectangle(img, (half_screen_width, 0), (1280, 720), (0, 255, 0), 2)  # Right box for X, Y, Z

        # Process right hand for servoL and servoR control
        if hand_r is not None:
            lmList_r = hand_r['lmList']  # List of 21 Landmark points
            
            # Green dot on the top of the right hand
            wrist_x, wrist_y = lmList_r[0][0], lmList_r[0][1]  # Wrist position
            cv2.circle(img, (int(wrist_x), int(wrist_y - 30)), 10, (0, 255, 0), cv2.FILLED)  # Dot above wrist

            x_pos = lmList_r[9][0]  # Using middle finger MCP joint for stability
            y_pos = lmList_r[9][1]

            if detector.fingersUp(hand_r) == [0, 0, 0, 0, 0]:  # All fingers down (fist grip)
                stop_servos()
                last_movement_time = time.time()
            else:
                if x_pos < half_screen_width:  # Left side of the screen
                    box_center_x = half_screen_width // 2
                    box_center_y = half_screen_height // 2

                    if y_pos < box_center_y - (box_center_y // 4):
                        move_servos(140,40)  # Move forward
                    elif y_pos > box_center_y + (box_center_y // 4):
                        move_servos(40, 140)  # Move backward
                    elif x_pos < box_center_x - (box_center_x // 4):
                        move_servos(neutral_position, 40)  # Turn left
                    elif x_pos > box_center_x + (box_center_x // 4):
                        move_servos(neutral_position, 140)  # Turn right
                    else:
                        stop_servos()
                    last_movement_time = time.time()
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
            servoX = np.interp(x_pos, [half_screen_width, 1280], [minDegX, maxDegX])
            barX = np.interp(x_pos, [half_screen_width, 1280], [25, 1255])
            
            # Y-axis control (vertical position)
            y_pos = lmList_l[9][1]
            servoY = np.interp(y_pos, [0, 720], [minDegY, maxDegY])
            barY = np.interp(y_pos, [0, 720], [25, 695])
            
            # Z-axis control (distance between thumb and index finger)
            length_z, info_z, img = detector.findDistance(lmList_l[4][0:2], lmList_l[8][0:2], img)
            servoZ = np.interp(length_z, [minHand, maxHand], [maxDegZ, minDegZ])
            barZ = np.interp(length_z, [minHand, maxHand], [600, 25])  # Inverse the bar for Z

            # Visual feedback for left hand controls
            posCircleX = int(np.interp(x_pos, [half_screen_width, 1280], [25, 1255]))
            posCircleY = int(np.interp(y_pos, [0, 720], [25, 695]))
            cv2.circle(img, (posCircleX, posCircleY), 25, (0, 0, 255), cv2.FILLED)

            # Horizontal bar and servo degree for X servo on the right side of the screen
            cv2.rectangle(img, (half_screen_width + 50, 650), (1280 - 50, 675), (255, 0, 0), 3)
            cv2.rectangle(img, (half_screen_width + 50, 650), (int(barX), 675), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoX)} deg', (half_screen_width + 50, 645), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            
            # Vertical bar and servo degree for Y servo on the right side of the screen
            cv2.rectangle(img, (half_screen_width + 25, 25), (half_screen_width + 50, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (half_screen_width + 25, int(barY)), (half_screen_width + 50, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoY)} deg', (half_screen_width + 55, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            
            # Vertical bar and servo degree for Z servo on the right side of the screen
            cv2.rectangle(img, (1205, 25), (1230, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (1205, int(barZ)), (1230, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoZ)} deg', (1050, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            
            # Send servo positions to Arduino
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)
            servo_pinZ.write(servoZ)
            
    # Show frame
    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
