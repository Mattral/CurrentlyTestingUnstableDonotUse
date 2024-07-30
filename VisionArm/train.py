import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pickle

def train_model():
    # Load data from CSV
    data = pd.read_csv('hand_servo_data.csv')

    # Extract relevant features (x, y, z positions and servo angles)
    X = data[['x', 'y', 'z']].values
    y = data[['servoX', 'servoY', 'servoZ']].values

    # Prepare sequences
    sequence_length = 10  # Number of previous points to consider
    future_steps = 1  # Number of future points to predict

    def create_sequences(data, target, sequence_length):
        X_seq, y_seq = [], []
        for i in range(len(data) - sequence_length):
            X_seq.append(data[i:i+sequence_length])
            y_seq.append(target[i+sequence_length])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X, y, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, X.shape[1])),
        Dense(50, activation='relu'),
        Dense(y.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    # Save the trained model
    with open('sequence_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved.")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pickle

def train_model():
    # Load data from CSV
    data = pd.read_csv('hand_servo_data.csv')

    # Extract relevant features (x, y, z positions and servo angles)
    X = data[['x', 'y', 'z']].values
    y = data[['servoX', 'servoY', 'servoZ']].values

    # Prepare sequences
    sequence_length = 10  # Number of previous points to consider
    future_steps = 1  # Number of future points to predict

    def create_sequences(data, target, sequence_length):
        X_seq, y_seq = [], []
        for i in range(len(data) - sequence_length):
            X_seq.append(data[i:i+sequence_length])
            y_seq.append(target[i+sequence_length])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X, y, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, X.shape[1])),
        Dense(50, activation='relu'),
        Dense(y.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    # Save the trained model
    with open('sequence_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved.")
