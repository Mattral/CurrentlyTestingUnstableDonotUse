import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyfirmata
import time
import threading
import speech_recognition as sr

# Initialize video capture
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize Arduino board
port = "COM6"  # Updated COM port
board = pyfirmata.Arduino(port)

# Initialize servos for robot arm (X, Y, Z)
servo_pinX = board.get_pin('d:3:s')
servo_pinY = board.get_pin('d:5:s')
servo_pinZ = board.get_pin('d:6:s')

# Initialize servos for movement (L and R)
servoL = board.get_pin('d:9:s')
servoR = board.get_pin('d:10:s')

# Initialize variables for robot arm
minHand, maxHand = 20, 220
minDegX, maxDegX = 60, 180
minDegY, maxDegY = 40, 140
minDegZ, maxDegZ = 100, 150
servoX = 120
servoY = 120
servoZ = 120

# Define the dimensions and positions of the boxes
center_x, center_y = 640, 360
half_screen_width = 640
half_screen_height = 360

# Initialize servo positions
neutral_position = 90
servoL_position = neutral_position
servoR_position = neutral_position

# Time tracking for movement stop
last_movement_time = time.time()

# Speech recognition setup
recognizer = sr.Recognizer()
microphone = sr.Microphone()

commands = {
    "forward": (40, 140),
    "go": (40, 140),
    "reverse": (140, 40),
    "back": (140, 40),
    "left": (40, 90),
    "laugh": (40, 90),
    "right": (90, 140)
}

def stop_servos():
    """Function to stop both movement servos."""
    servoL.write(neutral_position)
    servoR.write(neutral_position)

def move_servos(left_pos, right_pos):
    """Function to move both movement servos to specified positions."""
    servoL.write(left_pos)
    servoR.write(right_pos)

def speech_recognition_thread():
    """Thread function for handling speech recognition."""
    global last_movement_time
    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")

            if command in commands:
                servoL_angle, servoR_angle = commands[command]
                move_servos(servoL_angle, servoR_angle)
                last_movement_time = time.time()
            else:
                print("Unknown command")

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

def overlay_text(img, text):
    """Function to overlay text on the video feed."""
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Start speech recognition thread
speech_thread = threading.Thread(target=speech_recognition_thread, daemon=True)
speech_thread.start()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)

    if hands:
        # Identify right and left hands
        hand_r = next((h for h in hands if h['type'] == 'Right'), None)
        hand_l = next((h for h in hands if h['type'] == 'Left'), None)

        # Draw the control boxes
        cv2.rectangle(img, (0, 0), (half_screen_width, 720), (255, 0, 0), 2)
        cv2.rectangle(img, (half_screen_width, 0), (1280, 720), (0, 255, 0), 2)

        if hand_r is not None:
            lmList_r = hand_r['lmList']
            wrist_x, wrist_y = lmList_r[0][0], lmList_r[0][1]
            cv2.circle(img, (int(wrist_x), int(wrist_y - 30)), 10, (0, 255, 0), cv2.FILLED)

            x_pos = lmList_r[9][0]
            y_pos = lmList_r[9][1]

            if detector.fingersUp(hand_r) == [0, 0, 0, 0, 0]:
                stop_servos()
                last_movement_time = time.time()
            else:
                if x_pos < half_screen_width:
                    box_center_x = half_screen_width // 2
                    box_center_y = half_screen_height // 2

                    if y_pos < box_center_y - (box_center_y // 4):
                        move_servos(140, 40)
                    elif y_pos > box_center_y + (box_center_y // 4):
                        move_servos(40, 140)
                    elif x_pos < box_center_x - (box_center_x // 4):
                        move_servos(neutral_position, 40)
                    elif x_pos > box_center_x + (box_center_x // 4):
                        move_servos(neutral_position, 140)
                    else:
                        stop_servos()
                    last_movement_time = time.time()
                else:
                    stop_servos()
                    last_movement_time = time.time()

        if time.time() - last_movement_time >= 2:
            stop_servos()

        if hand_l is not None:
            lmList_l = hand_l['lmList']
            x_pos = lmList_l[9][0]
            servoX = np.interp(x_pos, [half_screen_width, 1280], [minDegX, maxDegX])
            barX = np.interp(x_pos, [half_screen_width, 1280], [25, 1255])

            y_pos = lmList_l[9][1]
            servoY = np.interp(y_pos, [0, 720], [minDegY, maxDegY])
            barY = np.interp(y_pos, [0, 720], [25, 695])

            length_z, info_z, img = detector.findDistance(lmList_l[4][0:2], lmList_l[8][0:2], img)
            servoZ = np.interp(length_z, [minHand, maxHand], [maxDegZ, minDegZ])
            barZ = np.interp(length_z, [minHand, maxHand], [600, 25])

            posCircleX = int(np.interp(x_pos, [half_screen_width, 1280], [25, 1255]))
            posCircleY = int(np.interp(y_pos, [0, 720], [25, 695]))
            cv2.circle(img, (posCircleX, posCircleY), 25, (0, 0, 255), cv2.FILLED)

            cv2.rectangle(img, (half_screen_width + 50, 650), (1280 - 50, 675), (255, 0, 0), 3)
            cv2.rectangle(img, (half_screen_width + 50, 650), (int(barX), 675), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoX)} deg', (half_screen_width + 50, 645), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            cv2.rectangle(img, (half_screen_width + 25, 25), (half_screen_width + 50, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (half_screen_width + 25, int(barY)), (half_screen_width + 50, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoY)} deg', (half_screen_width + 55, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            cv2.rectangle(img, (1205, 25), (1230, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (1205, int(barZ)), (1230, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoZ)} deg', (1050, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            servo_pinX.write(servoX)
            servo_pinY.write(servoY)
            servo_pinZ.write(servoZ)

    # Overlay the recognized speech on the video feed
    overlay_text(img, f"Servo L: {servoL_position} | Servo R: {servoR_position}")

    # Show the video feed
    cv2.imshow("Image", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
board.exit()
