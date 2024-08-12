// Define pin numbers
const int flameSensorPin = A0; // Analog pin for flame sensor
const int pirSensorPin = 2;    // Digital pin for PIR sensor
const int relayPin = 8;        // Digital pin for relay module
const int alarmPin = 9;        // Digital pin for buzzer/alarm

// Define threshold for flame detection
const int flameThreshold = 300; // Adjust this threshold based on your sensor

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Set pin modes
  pinMode(flameSensorPin, INPUT);
  pinMode(pirSensorPin, INPUT);
  pinMode(relayPin, OUTPUT);
  pinMode(alarmPin, OUTPUT);

  // Initially, turn off the relay and alarm
  digitalWrite(relayPin, LOW);
  digitalWrite(alarmPin, LOW);
}

void loop() {
  // Read the flame sensor value
  int flameValue = analogRead(flameSensorPin);
  
  // Read the PIR sensor value
  int pirValue = digitalRead(pirSensorPin);

  // Print sensor values to the serial monitor
  Serial.print("Flame Sensor Value: ");
  Serial.print(flameValue);
  Serial.print(" | PIR Sensor Value: ");
  Serial.println(pirValue);

  // Check for flame detection
  if (flameValue > flameThreshold) {
    digitalWrite(relayPin, HIGH); // Turn on the water pump
  } else {
    digitalWrite(relayPin, LOW);  // Turn off the water pump
  }

  // Check for human presence
  if (pirValue == HIGH) {
    digitalWrite(alarmPin, HIGH); // Turn on the alarm
  } else {
    digitalWrite(alarmPin, LOW);  // Turn off the alarm
  }

  // Delay to prevent excessive serial printing
  delay(200);
}
