#include <Servo.h>

/*
  Smart conveyor waste sorter
  - Board: Arduino Nano V3 (CH340G)
  - Sensor: E3F obstacle sensor (digital OUT)
  - Actuator: MG995/MG996 servo

  Protocol from Python over serial:
    C:R\n  -> recyclable
    C:N\n  -> non-recyclable
*/

Servo pusherServo;

const uint8_t PIN_SENSOR = 2;   // E3F OUT -> D2 (interrupt-capable)
const uint8_t PIN_SERVO = 9;    // Servo signal pin
const uint8_t PIN_LED = 13;     // Optional debug LED

// Adjust these angles to match your mechanical setup.
const int SERVO_IDLE_ANGLE = 90;
const int SERVO_RECYCLE_ANGLE = 35;
const int SERVO_NONREC_ANGLE = 145;

const unsigned long PUSH_HOLD_MS = 420;
const unsigned long RETURN_HOLD_MS = 280;
const unsigned long SENSOR_DEBOUNCE_MS = 80;

char pendingClass = 'N';  // Default to non-recyclable for safety.
unsigned long lastSensorTriggerMs = 0;

void moveServoTo(int angleDeg) {
  angleDeg = constrain(angleDeg, 0, 180);
  pusherServo.write(angleDeg);
}

void runPushCycle(char cls) {
  if (cls == 'R') {
    moveServoTo(SERVO_RECYCLE_ANGLE);
  } else {
    moveServoTo(SERVO_NONREC_ANGLE);
  }

  delay(PUSH_HOLD_MS);
  moveServoTo(SERVO_IDLE_ANGLE);
  delay(RETURN_HOLD_MS);
}

void parseSerialLine(String line) {
  line.trim();
  if (line.length() < 3) return;

  // Expect "C:R" or "C:N"
  if (line.charAt(0) == 'C' && line.charAt(1) == ':') {
    char cls = line.charAt(2);
    if (cls == 'R' || cls == 'N') {
      pendingClass = cls;
      Serial.print(F("ACK CLASS "));
      Serial.println(pendingClass);
      digitalWrite(PIN_LED, pendingClass == 'R' ? HIGH : LOW);
    }
  }
}

void readSerialCommands() {
  static String rx = "";
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rx.length() > 0) {
        parseSerialLine(rx);
        rx = "";
      }
    } else {
      if (rx.length() < 32) {
        rx += c;
      }
    }
  }
}

bool isObjectDetected() {
  // Most E3F NPN NO modules output LOW when object detected.
  return digitalRead(PIN_SENSOR) == LOW;
}

void setup() {
  pinMode(PIN_SENSOR, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  pusherServo.attach(PIN_SERVO);
  moveServoTo(SERVO_IDLE_ANGLE);

  Serial.begin(115200);
  Serial.println(F("SORTER READY"));
}

void loop() {
  readSerialCommands();

  unsigned long now = millis();
  if (isObjectDetected()) {
    if (now - lastSensorTriggerMs > SENSOR_DEBOUNCE_MS) {
      lastSensorTriggerMs = now;
      Serial.print(F("TRIGGER -> "));
      Serial.println(pendingClass);
      runPushCycle(pendingClass);
    }
  }
}
