import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import time
import pickle
import pyfirmata

# Initialize video capture with USB webcam index (e.g., 1)
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.9, maxHands=2)

# Initialize Arduino board
port = "COM5"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:5:s')  # pin 5 Arduino
servo_pinY = board.get_pin('d:6:s')  # pin 6 Arduino
servo_pinZ = board.get_pin('d:9:s')  # pin 9 Arduino

# Initialize variables
minHand, maxHand = 20, 220

# Set thresholds for each servo
minDegX, maxDegX = 45, 135  # Adjust these values for X servo
minDegY, maxDegY = 30, 150  # Adjust these values for Y servo
minDegZ, maxDegZ = 100, 130  # Adjust these values for Z servo

servoX = 90  # Starting position
servoY = 90  # Starting position
servoZ = 90  # Starting position

# Button properties
button_color = (0, 255, 0)
button_hover_color = (0, 200, 0)
button_width, button_height = 200, 50

# Button positions
buttons = {
    "Collect Data": (460, 20),
    "Stop Collecting": (680, 20),
    "Train Model": (900, 20),
    "AI Mode": (1120, 20)
}

# Initialize flags and variables
collect_data = False
train_model = False
ai_mode = False
countdown = False
data = []
countdown_time = 3  # Countdown time in seconds
collect_start_time = None
notification = ""
notification_start_time = None
models = None  # For storing the trained models
predicted_movements = []  # Store the series of predicted movements
movement_sequence = []  # Store movement sequence for replaying

# Function to draw buttons
def draw_buttons(img):
    for key, (x, y) in buttons.items():
        color = button_hover_color if (ai_mode and key == "AI Mode") else button_color
        cv2.rectangle(img, (x, y), (x + button_width, y + button_height), color, cv2.FILLED)
        button_text = key if not (collect_data and key == "Collect Data") else "Stop Collecting"
        cv2.putText(img, button_text, (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Function to check if a point is inside a rectangle
def is_inside(x, y, rect_x, rect_y, rect_w, rect_h):
    return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h

# Function to save data to CSV
def save_data_to_csv():
    global data
    with open('hand_servo_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'z', 'servoX', 'servoY', 'servoZ'])
        writer.writerows(data)
    data = []  # Clear the data list after saving
    print("Data saved to hand_servo_data.csv")

# Function to load trained models
def load_models():
    global models
    try:
        with open('servo_models.pkl', 'rb') as file:
            models = pickle.load(file)
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Trained models not found. Train the model first.")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load the trained models at the start
load_models()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)  # Update findHands call to return both hands and image

    # Draw the buttons
    draw_buttons(img)

    if hands:
        # Check for right hand
        hand_r = None
        if hands[0]['type'] == 'Right':
            hand_r = hands[0]
        elif len(hands) == 2 and hands[1]['type'] == 'Right':
            hand_r = hands[1]

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
            servoZ = np.interp(length_z, [minHand, maxHand], [minDegZ, maxDegZ])
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

            # Update servos with current positions
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)
            servo_pinZ.write(servoZ)

            # Check if index finger tip is clicking a button
            finger_tip = lmList_r[8]  # Index finger tip
            for key, (x, y) in buttons.items():
                if is_inside(finger_tip[0], finger_tip[1], x, y, button_width, button_height):
                    cv2.rectangle(img, (x, y), (x + button_width, y + button_height), button_hover_color, cv2.FILLED)
                    cv2.putText(img, key, (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Detect click (proximity to the button center)
                    if abs(finger_tip[0] - (x + button_width // 2)) < 20 and abs(finger_tip[1] - (y + button_height // 2)) < 20:
                        if key == "Collect Data":
                            if not collect_data:
                                collect_data = True
                                countdown = True
                                collect_start_time = time.time()
                                notification = "Data collection started"
                            else:
                                collect_data = False
                                save_data_to_csv()
                                notification = "Data collection stopped"
                        elif key == "Stop Collecting":
                            collect_data = False
                            notification = "Data collection stopped"
                            save_data_to_csv()
                        elif key == "Train Model":
                            if data:
                                # Replace train.main() with actual model training
                                # train.main()  
                                load_models()  # Reload the models after training
                                notification = "Model training completed"
                            else:
                                notification = "No data collected for training"
                        elif key == "AI Mode":
                            ai_mode = not ai_mode
                            notification = "AI Mode " + ("enabled" if ai_mode else "disabled")
                            if ai_mode:
                                # Start replaying movements
                                movement_sequence = data[:]
                                print("Replaying movements")
                            notification_start_time = time.time()

    # Handle countdown for data collection
    if countdown:
        elapsed_time = time.time() - collect_start_time
        countdown_left = countdown_time - int(elapsed_time)
        if countdown_left <= 0:
            countdown = False
            notification = "Data collection in progress"
            notification_start_time = time.time()
        else:
            cv2.putText(img, f'Starting in {countdown_left}', (520, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Collect data
    if collect_data and not countdown:
        data.append([servoX, servoY, servoZ])
        cv2.putText(img, f'Collecting Data: {len(data)} samples', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display notification
    if notification:
        if notification_start_time and time.time() - notification_start_time < 3:  # Display notification for 3 seconds
            cv2.putText(img, notification, (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            notification = ""

    # AI mode handling
    if ai_mode:
        if movement_sequence:
            if len(predicted_movements) < len(movement_sequence):
                # Replay the recorded movements
                movement = movement_sequence[len(predicted_movements)]
                predicted_movements.append(movement)

                # Set servo positions to the replayed values
                servoX, servoY, servoZ = movement
                servo_pinX.write(servoX)
                servo_pinY.write(servoY)
                servo_pinZ.write(servoZ)

                # Draw blue dots on the image for each movement
                posCircleX = int(np.interp(servoX, [minDegX, maxDegX], [25, 1255]))
                posCircleY = int(np.interp(servoY, [minDegY, maxDegY], [25, 695]))
                cv2.circle(img, (posCircleX, posCircleY), 10, (255, 0, 0), cv2.FILLED)
            else:
                cv2.putText(img, 'Replay Completed', (10, 670), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(img, 'No Data to Replay', (10, 670), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the image
    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
