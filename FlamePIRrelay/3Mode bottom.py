import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import time
import train  # Import the train module
import pickle
import pyfirmata

# Initialize video capture with USB webcam index (e.g., 1)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.9, maxHands=2)
'''
# Initialize Arduino board
port = "COM5"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:5:s')  # pin 5 Arduino
servo_pinY = board.get_pin('d:6:s')  # pin 6 Arduino
servo_pinZ = board.get_pin('d:9:s')  # pin 9 Arduino
'''

# Initialize variables
minHand, maxHand = 20, 220

# Set thresholds for each servo
minDegX, maxDegX = 45, 135  # Adjust these values for X servo
minDegY, maxDegY = 30, 150  # Adjust these values for Y servo
minDegZ, maxDegZ = 100, 130  # Adjust these values for Z servo

servoX = 90  # Starting position
servoY = 90  # Starting positio
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
                            else:
                                collect_data = False
                                save_data_to_csv()
                                notification = "CSV saved"
                                notification_start_time = time.time()
                        elif key == "Train Model":
                            train.train_model()  # Call the function from train module
                            notification = "Model trained and saved"
                            notification_start_time = time.time()
                        elif key == "AI Mode":
                            print("AI bottom clicked")
                            ai_mode = not ai_mode  # Toggle AI mode

    # Handle data collection
    if collect_data and not countdown:
        current_time = time.time()
        if current_time - collect_start_time < 30:  # Collect data for 30 seconds
            data.append([x_pos, y_pos, length_z, servoX, servoY, servoZ])
        else:
            collect_data = False
            save_data_to_csv()
            notification = "CSV saved"
            notification_start_time = time.time()

    # Handle countdown for data collection
    if countdown:
        elapsed_time = time.time() - collect_start_time
        if elapsed_time < countdown_time:
            cv2.putText(img, f"Starting in {int(countdown_time - elapsed_time)}...", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
        else:
            collect_data = True
            countdown = False
            notification = "Collecting data..."
            notification_start_time = time.time()
            collect_start_time = time.time()  # Reset the collection start time for actual data collection


    if ai_mode and models:
        if not predicted_movements:
            # Generate predicted movements for the next 10 seconds
            for t in range(1, 101):  # 10 seconds at 10 predictions per second
                future_x_pos = x_pos + t * 10  # Replace with actual model logic
                future_y_pos = y_pos + t * 10  # Replace with actual model logic
                future_length_z = length_z + t * 1  # Replace with actual model logic
                X = np.array([[future_x_pos, future_y_pos, future_length_z]])
                predicted_servos = [model.predict(X)[0] for model in models]
                predicted_movements.append(predicted_servos)
            print("Starting predictions")

        # Display AI predicted positions (blue pointer)
        if predicted_movements:
            ai_servoX, ai_servoY, ai_servoZ = predicted_movements.pop(0)
            ai_posCircleX = int(np.interp(ai_servoX, [minDegX, maxDegX], [25, 1255]))
            ai_posCircleY = int(np.interp(ai_servoY, [minDegY, maxDegY], [25, 695]))
            cv2.circle(img, (ai_posCircleX, ai_posCircleY), 25, (255, 0, 0), cv2.FILLED)

            # Draw horizontal bar for AI predicted X servo position
            ai_barX = np.interp(ai_servoX, [minDegX, maxDegX], [25, 1255])
            cv2.rectangle(img, (50, 680), (int(ai_barX), 705), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, f'AI {int(ai_servoX)} deg', (50, 675), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

            # Draw vertical bar for AI predicted Y servo position
            ai_barY = np.interp(ai_servoY, [minDegY, maxDegY], [25, 600])
            cv2.rectangle(img, (50, int(ai_barY)), (75, 600), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, f'AI {int(ai_servoY)} deg', (80, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

            # Draw vertical bar for AI predicted Z servo position
            ai_barZ = np.interp(ai_servoZ, [minDegZ, maxDegZ], [600, 25])
            cv2.rectangle(img, (1205, int(ai_barZ)), (1230, 600), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, f'AI {int(ai_servoZ)} deg', (1050, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

            print("Showing predictions")
        else:
            ai_mode = False  # Turn off AI mode if no more predictions are left


    # Display notification if any
    if notification:
        cv2.putText(img, notification, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if time.time() - notification_start_time > 1:  # Display notification for 1 second
            notification = ""

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import time
import train  # Import the train module
import pickle

# Initialize video capture with USB webcam index (e.g., 1)
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize variables
minHand, maxHand = 20, 220

# Set thresholds for each servo
minDegX, maxDegX = 45, 135  # Adjust these values for X servo
minDegY, maxDegY = 30, 150  # Adjust these values for Y servo
minDegZ, maxDegZ = 60, 120  # Adjust these values for Z servo

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
                            else:
                                collect_data = False
                                save_data_to_csv()
                                notification = "CSV saved"
                                notification_start_time = time.time()
                        elif key == "Train Model":
                            train.train_model()  # Train the model
                            notification = "Model trained and saved"
                            notification_start_time = time.time()
                        elif key == "AI Mode":
                            ai_mode = not ai_mode  # Toggle AI mode

    # Handle data collection
    if collect_data and not countdown:
        current_time = time.time()
        if current_time - collect_start_time < 10:  # Collect data for 10 seconds
            data.append([x_pos, y_pos, length_z, servoX, servoY, servoZ])
        else:
            collect_data = False
            save_data_to_csv()
            notification = "CSV saved"
            notification_start_time = time.time()

    # Handle countdown for data collection
    if countdown:
        elapsed_time = time.time() - collect_start_time
        if elapsed_time < countdown_time:
            cv2.putText(img, f"Starting in {int(countdown_time - elapsed_time)}...", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
        else:
            collect_data = True
            countdown = False
            notification = "Collecting data..."
            notification_start_time = time.time()
            collect_start_time = time.time()  # Reset the collection start time for actual data collection
    '''
    # Handle AI mode predictions and execution
    if ai_mode and models:
        if not predicted_movements:
            # Generate predicted movements for the next 10 seconds
            for t in range(1, 101):  # 10 seconds at 10 predictions per second
                future_x_pos = x_pos + t * 10  # Replace with actual model logic
                future_y_pos = y_pos + t * 10  # Replace with actual model logic
                future_length_z = length_z + t * 1  # Replace with actual model logic
                X = np.array([[future_x_pos, future_y_pos, future_length_z]])
                predicted_servos = [model.predict(X)[0] for model in models]
                predicted_movements.append(predicted_servos)
                print("starting predictions")
        
        # Display AI predicted positions (blue pointer)
        if predicted_movements:
            ai_servoX, ai_servoY, ai_servoZ = predicted_movements.pop(0)
            ai_posCircleX = int(np.interp(ai_servoX, [minDegX, maxDegX], [25, 1255]))
            ai_posCircleY = int(np.interp(ai_servoY, [minDegY, maxDegY], [25, 695]))
            cv2.circle(img, (ai_posCircleX, ai_posCircleY), 25, (255, 0, 0), cv2.FILLED)
            print("showing predictions")
        else:
            ai_mode = False  # Turn off AI mode if no more predictions are left
    '''



    # Display notification if any
    if notification:
        cv2.putText(img, notification, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if time.time() - notification_start_time > 1:  # Display notification for 1 second
            notification = ""

    # Handle AI mode with timer logic
    if ai_mode and models:
        # Perform AI mode predictions and control servos accordingly
        if hands:
            if hand_r is not None:
                # Create the feature vector for prediction
                features = np.array([x_pos, y_pos, length_z]).reshape(1, -1)
                # Predict servo values using the trained models
                predicted_servoX = models['servoX_model'].predict(features)[0]
                predicted_servoY = models['servoY_model'].predict(features)[0]
                predicted_servoZ = models['servoZ_model'].predict(features)[0]
                
                # Update servos with predicted values
                servoX = np.clip(predicted_servoX, minDegX, maxDegX)
                servoY = np.clip(predicted_servoY, minDegY, maxDegY)
                servoZ = np.clip(predicted_servoZ, minDegZ, maxDegZ)
                
                # Store the predicted movements for smooth control
                predicted_movements.append((servoX, servoY, servoZ))
                if len(predicted_movements) > 10:
                    predicted_movements.pop(0)

                # Display the AI mode status and countdown
                cv2.putText(img, 'AI Mode Active', (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check if 30 seconds have passed since AI mode started
        if time.time() - ai_mode_start_time >= 30:
            ai_mode = False
            cv2.putText(img, 'AI Mode Stopped', (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    '''
    # Send the servo values to the Arduino
    servo_pinX.write(servoX)
    servo_pinY.write(servoY)
    servo_pinZ.write(servoZ)
    '''


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
