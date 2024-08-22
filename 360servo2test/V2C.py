import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyfirmata
import time
import threading
import speech_recognition as sr
import pyttsx3  # Text-to-speech library

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize Arduino board
port = "COM6"  # Update COM port if necessary
board = pyfirmata.Arduino(port)

# Initialize servos for robot arm (X, Y, Z)
servo_pinX = board.get_pin('d:3:s')
servo_pinY = board.get_pin('d:5:s')
servo_pinZ = board.get_pin('d:6:s')

# Initialize L298N motor driver pins
left_motor_dir1 = board.get_pin('d:7:o')
left_motor_dir2 = board.get_pin('d:8:o')
left_motor_pwm = board.get_pin('d:9:p')

right_motor_dir1 = board.get_pin('d:13:o')
right_motor_dir2 = board.get_pin('d:12:o')
right_motor_pwm = board.get_pin('d:10:p')

# Initialize servos for legs (knee and ankle)
knee_left_servo = board.get_pin('a:0:s')
ankle_left_servo = board.get_pin('a:1:s')
knee_right_servo = board.get_pin('a:2:s')
ankle_right_servo = board.get_pin('a:3:s')

# Define positions for "Sit", "Bend", and "Stand"
positions = {
    "sit": [30, 90, 30, 90],   # [knee_left, ankle_left, knee_right, ankle_right]
    "bend": [60, 120, 60, 120],
    "stand": [90, 150, 90, 150]
}

# Speech synthesis setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate

# Initialize variables for robot arm
minHand, maxHand = 20, 220
minDegX, maxDegX = 60, 180
minDegY, maxDegY = 40, 140
minDegZ, maxDegZ = 100, 150
servoX = 120
servoY = 120
servoZ = 120

# Initialize motor control variables
neutral_speed = 0.0
max_speed = 1.0
min_speed = 0.5

# Initialize data collection variables
collect_data = False
data = []

# Speech recognition setup
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# AI Mode
ai_mode = False
models = None

# Commands for speech recognition
commands = {
    "forward": (1, 1, max_speed, max_speed),
    "go": (1, 1, max_speed, max_speed),
    "reverse": (0, 0, max_speed, max_speed),
    "back": (0, 0, max_speed, max_speed),
    "left": (0, 1, max_speed, max_speed),
    "right": (1, 0, max_speed, max_speed),
    "sit": "sit",
    "bend": "bend",
    "stand": "stand",
    "walk": "walk"
}

def stop_motors():
    """Function to stop both motors."""
    left_motor_pwm.write(neutral_speed)
    right_motor_pwm.write(neutral_speed)

def move_motors(left_dir1, left_dir2, left_speed, right_dir1, right_dir2, right_speed):
    """Function to move both motors with specified directions and speeds."""
    left_motor_dir1.write(left_dir1)
    left_motor_dir2.write(left_dir2)
    left_motor_pwm.write(left_speed)
    
    right_motor_dir1.write(right_dir1)
    right_motor_dir2.write(right_dir2)
    right_motor_pwm.write(right_speed)

def move_servos(position):
    """Function to move the leg servos to a specific position."""
    knee_left_servo.write(position[0])
    ankle_left_servo.write(position[1])
    knee_right_servo.write(position[2])
    ankle_right_servo.write(position[3])

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
                if command in ["sit", "bend", "stand"]:
                    move_servos(positions[command])
                    speak_command(command)
                elif command == "walk":
                    walk_sequence()
                    speak_command("Walking")
                else:
                    left_dir, right_dir, left_speed, right_speed = commands[command]
                    move_motors(left_dir, not left_dir, left_speed, right_dir, not right_dir, right_speed)
                    last_movement_time = time.time()
                    speak_command(command)
            else:
                print("Unknown command")

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

def walk_sequence():
    """Function to mimic walking by moving leg servos in sequence."""
    walk_positions = [
        [90, 150, 30, 90],   # Left forward, right backward
        [30, 90, 90, 150],   # Right forward, left backward
    ]
    for _ in range(2):  # Two loops to mimic walking
        for pos in walk_positions:
            move_servos(pos)
            speak_command("Moving")
            time.sleep(0.5)  # Pause to allow servos to move

def speak_command(command):
    """Function to provide verbal feedback on commands."""
    engine.say(f"Executing {command}")
    engine.runAndWait()

def overlay_text(img, text):
    """Function to overlay text on the video feed."""
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

def button_click(event, x, y, flags, param):
    """Function to handle button clicks."""
    global collect_data, ai_mode, models
    if event == cv2.EVENT_LBUTTONDOWN:
        if 50 <= x <= 250 and 50 <= y <= 100:
            print("Data Collection Button Clicked")
            collect_data = not collect_data
            if not collect_data:
                save_data_to_csv()
        elif 300 <= x <= 500 and 50 <= y <= 100:
            print("Training Button Clicked")
            save_data_to_csv()
            train.train_model()
            models = train.load_models()
        elif 550 <= x <= 750 and 50 <= y <= 100:
            print("AI Button Clicked")
            ai_mode = not ai_mode
        elif 800 <= x <= 1000 and 50 <= y <= 100:
            print("Sit Button Clicked")
            move_servos(positions["sit"])
            speak_command("sit")
        elif 1050 <= x <= 1250 and 50 <= y <= 100:
            print("Bend Button Clicked")
            move_servos(positions["bend"])
            speak_command("bend")
        elif 1300 <= x <= 1500 and 50 <= y <= 100:
            print("Stand Button Clicked")
            move_servos(positions["stand"])
            speak_command("stand")

def save_data_to_csv():
    """Function to save collected data to a CSV file."""
    global data
    if data:
        np.savetxt('hand_data.csv', np.array(data), delimiter=',', fmt='%f')
        print(f'Data saved to hand_data.csv with {len(data)} entries.')
        data = []

# Start speech recognition thread
speech_thread = threading.Thread(target=speech_recognition_thread, daemon=True)
speech_thread.start()

cv2.namedWindow("Hand Tracking")
cv2.setMouseCallback("Hand Tracking", button_click)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)

    # Draw the buttons
    cv2.rectangle(img, (50, 50), (250, 100), (0, 255, 0), -1)  # Data Collection Button
    cv2.putText(img, "Data Collect", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(img, (300, 50), (500, 100), (255, 0, 0), -1)  # Training Button
    cv2.putText(img, "Train", (340, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(img, (550, 50), (750, 100), (0, 0, 255), -1)  # AI Mode Button
    cv2.putText(img, "AI Mode", (570, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(img, (800, 50), (1000, 100), (255, 255, 0), -1)  # Sit Button
    cv2.putText(img, "Sit", (860, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(img, (1050, 50), (1250, 100), (0, 255, 255), -1)  # Bend Button
    cv2.putText(img, "Bend", (1100, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(img, (1300, 50), (1500, 100), (255, 0, 255), -1)  # Stand Button
    cv2.putText(img, "Stand", (1360, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 Landmark points
        bbox = hand["bbox"]      # Bounding box info x,y,w,h
        centerPoint = hand['center']  # center of the hand cx,cy

        if len(lmList) != 0:
            x, y, z = lmList[8][0:3]
            servoX = np.interp(x, [minHand, maxHand], [minDegX, maxDegX])
            servoY = np.interp(y, [minHand, maxHand], [minDegY, maxDegY])
            servoZ = np.interp(z, [minHand, maxHand], [minDegZ, maxDegZ])

            servo_pinX.write(servoX)
            servo_pinY.write(servoY)
            servo_pinZ.write(servoZ)

            # Data collection logic
            if collect_data:
                data.append([servoX, servoY, servoZ])
                overlay_text(img, "Collecting Data...")

        if ai_mode and models:
            # Placeholder for AI control logic
            overlay_text(img, "AI Mode Active")

    cv2.imshow("Hand Tracking", img)

    # Stop motors after 1 second of no command
    if time.time() - last_movement_time > 1:
        stop_motors()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
