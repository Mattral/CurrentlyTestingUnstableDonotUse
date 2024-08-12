import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyfirmata
import time
import speech_recognition as sr
import threading

# Initialize video capture with USB webcam index (e.g., 1)
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize Arduino board
port = "COM6"
board = pyfirmata.Arduino(port)

# Initialize servos for hand tracking
servo_pinX = board.get_pin('d:3:s')
servo_pinY = board.get_pin('d:5:s')
servo_pinZ = board.get_pin('d:6:s')

# Initialize servos for speech recognition
servo_pinL = board.get_pin('d:9:s')  # Left servo
servo_pinR = board.get_pin('d:10:s')  # Right servo

# Set initial position of servos for speech recognition
servo_pinL.write(90)
servo_pinR.write(90)

# Initialize variables for hand tracking
minHand, maxHand = 20, 220
minDegX, maxDegX = 60, 180
minDegY, maxDegY = 40, 140
minDegZ, maxDegZ = 100, 150
servoX, servoY, servoZ = 120, 120, 120

# Set up speech recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()
commands = {
    "forward": (0, 180),
    "go": (0, 180),
    "reverse": (180, 0),
    "back": (180, 0),
    "left": (0, 90),
    "laugh": (0, 90),
    "right": (90, 180)
}
current_command = ""

def speech_recognition_thread():
    global current_command
    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            if command in commands:
                servoL_angle, servoR_angle = commands[command]
                move_servos(servoL_angle, servoR_angle, 3)
                current_command = f"Command: {command.capitalize()}"
            else:
                # Unknown command: reset servos to 90
                servo_pinL.write(90)
                servo_pinR.write(90)
                current_command = "Unknown command"
        except sr.UnknownValueError:
            # Could not understand audio: reset servos to 90
            servo_pinL.write(90)
            servo_pinR.write(90)
            current_command = "Could not understand audio"
        except sr.RequestError as e:
            # Request error: reset servos to 90
            servo_pinL.write(90)
            servo_pinR.write(90)
            current_command = f"Request error: {e}"

def move_servos(servoL_angle, servoR_angle, duration):
    servo_pinL.write(servoL_angle)
    servo_pinR.write(servoR_angle)
    time.sleep(duration)
    stop_servos()

def stop_servos():
    servo_pinL.write(90)
    servo_pinR.write(90)

# Start speech recognition thread
thread = threading.Thread(target=speech_recognition_thread)
thread.daemon = True
thread.start()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)

    if hands:
        if hands[0]['type'] == 'Right':
            hand_r = hands[0]
        elif len(hands) == 2 and hands[1]['type'] == 'Right':
            hand_r = hands[1]
        else:
            hand_r = None

        if hand_r is not None:
            lmList_r = hand_r['lmList']
            x_pos = lmList_r[9][0]
            servoX = np.interp(x_pos, [0, 1280], [minDegX, maxDegX])
            barX = np.interp(x_pos, [0, 1280], [25, 1255])
            
            y_pos = lmList_r[9][1]
            servoY = np.interp(y_pos, [0, 720], [minDegY, maxDegY])
            barY = np.interp(y_pos, [0, 720], [25, 600])
            
            length_z, info_z, img = detector.findDistance(lmList_r[4][0:2], lmList_r[8][0:2], img)
            servoZ = np.interp(length_z, [minHand, maxHand], [maxDegZ, minDegZ])
            barZ = np.interp(length_z, [minHand, maxHand], [600, 25])

            posCircleX = int(np.interp(x_pos, [0, 1280], [25, 1255]))
            posCircleY = int(np.interp(y_pos, [0, 720], [25, 695]))
            cv2.circle(img, (posCircleX, posCircleY), 25, (0, 0, 255), cv2.FILLED)

            cv2.rectangle(img, (50, 650), (1230, 675), (255, 0, 0), 3)
            cv2.rectangle(img, (50, 650), (int(barX), 675), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoX)} deg', (50, 645), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            
            cv2.rectangle(img, (25, 25), (50, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (25, int(barY)), (50, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoY)} deg', (55, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            
            cv2.rectangle(img, (1205, 25), (1230, 600), (255, 0, 0), 3)
            cv2.rectangle(img, (1205, int(barZ)), (1230, 600), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(servoZ)} deg', (1050, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            servo_pinX.write(servoX)
            servo_pinY.write(servoY)
            servo_pinZ.write(servoZ)

    # Display the current command in the middle of the video
    height, width, _ = img.shape
    text_size = cv2.getTextSize(current_command, cv2.FONT_HERSHEY_PLAIN, 2, 3)[0]
    text_x = int((width - text_size[0]) / 2)
    text_y = int((height + text_size[1]) / 2)
    cv2.putText(img, current_command, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
