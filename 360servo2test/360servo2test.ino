#include <Servo.h>

Servo servo1;  // Create servo object for the first servo
Servo servo2;  // Create servo object for the second servo

void setup() {
  servo1.attach(10);  // Attach the first servo to pin 10
  servo2.attach(9);   // Attach the second servo to pin 9
}

void loop() {
  // Move forward for 3 seconds
  servo1.write(0);    // Rotate servo1 clockwise
  servo2.write(180);  // Rotate servo2 counterclockwise
  delay(3000);        // Move forward for 3 seconds

  // Stop both servos for 1 second
  servo1.write(90);   // Stop servo1
  servo2.write(90);   // Stop servo2
  delay(1000);        // Wait for 1 second

  // Move backward for 3 seconds
  servo1.write(180);  // Rotate servo1 counterclockwise
  servo2.write(0);    // Rotate servo2 clockwise
  delay(3000);        // Move backward for 3 seconds

  // Stop both servos for 1 second
  servo1.write(90);   // Stop servo1
  servo2.write(90);   // Stop servo2
  delay(1000);        // Wait for 1 second

  // Turn left for 3 seconds
  servo1.write(0);    // Rotate servo1 clockwise
  servo2.write(90);   // Stop servo2 (no movement)
  delay(3000);        // Turn left for 3 seconds

  // Stop both servos for 1 second
  servo1.write(90);   // Stop servo1
  servo2.write(90);   // Stop servo2
  delay(1000);        // Wait for 1 second

  // Turn right for 3 seconds
  servo1.write(90);   // Stop servo1 (no movement)
  servo2.write(180);  // Rotate servo2 counterclockwise
  delay(3000);        // Turn right for 3 seconds

  // Stop both servos for 1 second
  servo1.write(90);   // Stop servo1
  servo2.write(90);   // Stop servo2
  delay(1000);        // Wait for 1 second
}
