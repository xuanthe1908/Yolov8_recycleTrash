#if !defined(ESP8266)
#error "Select an ESP8266 board in Arduino IDE (e.g. NodeMCU 1.0, LOLIN WEMOS D1 mini)."
#endif

#include <Servo.h>

/*
  Smart conveyor waste sorter
  - Board: ESP8266 (NodeMCU 1.0, Wemos D1 mini, etc.)
  - Sensor: E3F obstacle sensor (digital OUT)
  - Actuator: MG995/MG996 servo (5 V external supply + common GND)

  Serial (USB), 115200 baud — same protocol as Python predict.py:
    MCU -> PC: TRIGGER + newline when E3F sees an object
    PC  -> MCU: C:R / C:N + newline (recyclable / non-recyclable)

  ESP8266 GPIO is 3.3 V only: E3F OUT must not drive 5 V into the pin
  (use 3.3 V pull-up, level shifter, or PNP/NPN wiring per datasheet).

  Pin map (NodeMCU / Wemos D1 mini silkscreen):
    PIN_SENSOR = GPIO4  -> labeled D2
    PIN_SERVO  = GPIO14 -> labeled D5 (PWM-friendly)
    PIN_LED    = GPIO2  -> labeled D4 (built-in LED on many boards; active LOW)
*/

Servo pusherServo;

const uint8_t PIN_SENSOR = 4;   // D2: E3F OUT (active LOW = object)
const uint8_t PIN_SERVO = 14;   // D5: servo signal
const uint8_t PIN_LED = 2;      // D4: status (optional)

const int SERVO_IDLE_ANGLE = 90;
const int SERVO_RECYCLE_ANGLE = 35;
const int SERVO_NONREC_ANGLE = 145;

const unsigned long PUSH_HOLD_MS = 420;
const unsigned long RETURN_HOLD_MS = 280;
const unsigned long SENSOR_DEBOUNCE_MS = 80;
const unsigned long CLASS_WAIT_TIMEOUT_MS = 450;

char pendingClass = 'N';
bool classUpdated = false;
unsigned long lastSensorTriggerMs = 0;

static void delayYield(unsigned long ms) {
  unsigned long t0 = millis();
  while (millis() - t0 < ms) {
    yield();
    delay(1);
  }
}

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

  delayYield(PUSH_HOLD_MS);
  moveServoTo(SERVO_IDLE_ANGLE);
  delayYield(RETURN_HOLD_MS);
}

void parseSerialLine(String &line) {
  line.trim();
  if (line.length() < 3) {
    return;
  }

  if (line.charAt(0) == 'C' && line.charAt(1) == ':') {
    char cls = line.charAt(2);
    if (cls == 'R' || cls == 'N') {
      pendingClass = cls;
      classUpdated = true;
      Serial.print(F("ACK CLASS "));
      Serial.println(pendingClass);
      // Built-in LED on GPIO2 is often active LOW.
      digitalWrite(PIN_LED, pendingClass == 'R' ? LOW : HIGH);
    }
  }
}

void readSerialCommands() {
  static String rx;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rx.length() > 0) {
        parseSerialLine(rx);
        rx = "";
      }
    } else if (rx.length() < 32) {
      rx += c;
    }
  }
}

bool isObjectDetected() {
  return digitalRead(PIN_SENSOR) == LOW;
}

void setup() {
  pinMode(PIN_SENSOR, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, HIGH);  // LED off if active LOW

  pusherServo.attach(PIN_SERVO);
  moveServoTo(SERVO_IDLE_ANGLE);

  Serial.begin(115200);
  Serial.println();
  Serial.println(F("SORTER READY (ESP8266)"));
}

char waitClassAfterTrigger() {
  classUpdated = false;
  unsigned long start = millis();
  while (millis() - start < CLASS_WAIT_TIMEOUT_MS) {
    readSerialCommands();
    if (classUpdated) {
      return pendingClass;
    }
    yield();
  }
  return pendingClass;
}

void loop() {
  readSerialCommands();

  unsigned long now = millis();
  if (isObjectDetected()) {
    if (now - lastSensorTriggerMs > SENSOR_DEBOUNCE_MS) {
      lastSensorTriggerMs = now;
      Serial.println(F("TRIGGER"));
      char cls = waitClassAfterTrigger();
      Serial.print(F("PUSH -> "));
      Serial.println(cls);
      runPushCycle(cls);
    }
  }
}
