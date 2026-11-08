/*
 * pca9685_servo_bridge.ino  (rewritten to match send_gestures_to_hardware.py)
 *
 * Receives one line per pose over USB serial and drives 16 servos through a
 * PCA9685 board over I2C.
 *
 * Wire format (from the Python side):
 *   "a0,a1,a2,...,a15\n"   16 angles in degrees, e.g. "90.0,90.0,...,90.0"
 *   value i  ->  PCA9685 channel i  (order = CHANNEL_ORDER in the Python file)
 *
 * Channel map (4 servos per finger: base joint, then Joint1..Joint3):
 *   0-3    Thumb   Revolute-2,  ThumbJoint1-3
 *   4-7    Index   Revolute-4,  IndexJoint1-3
 *   8-11   Middle  Revolute-6,  MiddleJoint1-3
 *   12-15  Pinky   Revolute-8,  PinkyJoint1-3
 *
 * Reply after every valid line:  "OK\n"
 * Reply for a malformed line:    "ERR <reason>\n"
 *
 * Library: "Adafruit PWM Servo Driver Library" (Library Manager).
 * Wiring:  Arduino SDA/SCL -> PCA9685 SDA/SCL, VCC -> 5V (logic), GND shared.
 *          Servo power goes to the PCA9685 V+ terminal from its OWN 5-6V
 *          supply, NOT from the Arduino 5V pin. Share ground with the Arduino.
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ---- Configuration ---------------------------------------------------------
const uint32_t BAUD_RATE      = 115200;   // must match BAUD_RATE in Python
const uint8_t  NUM_CHANNELS   = 16;
const uint8_t  PCA9685_ADDR   = 0x40;     // default I2C address
const float    PWM_FREQ_HZ    = 50.0;     // standard hobby-servo frame rate

// Pulse width at 0 and 180 degrees. 2g micro servos are typically around
// 500-2400 us, but this varies by model: calibrate yours. Wrong endpoints
// cause buzzing/stalling at the extremes.
const uint16_t PULSE_US_AT_0   = 500;
const uint16_t PULSE_US_AT_180 = 2400;

// Safety clamp on commanded angles. Driving a micro servo into its hard stop
// at 0 or 180 can stall and burn it out, so stay a little inside the range.
// (Python maps sim +/-90 deg onto 0..180, so this trims the extreme ends.)
const float SAFE_MIN_DEG = 10.0;
const float SAFE_MAX_DEG = 170.0;

// Longest valid line is about 16 * "180.0," + newline = ~97 bytes.
const size_t LINE_BUF_SIZE = 160;

// ---- State -----------------------------------------------------------------
Adafruit_PWMServoDriver pwm(PCA9685_ADDR);

char   lineBuf[LINE_BUF_SIZE];
size_t lineLen   = 0;
bool   overflow  = false;

// ---- Helpers ---------------------------------------------------------------
void setServoAngle(uint8_t channel, float deg) {
  if (deg < SAFE_MIN_DEG) deg = SAFE_MIN_DEG;
  if (deg > SAFE_MAX_DEG) deg = SAFE_MAX_DEG;
  float us = PULSE_US_AT_0 + (deg / 180.0f) * (PULSE_US_AT_180 - PULSE_US_AT_0);
  pwm.writeMicroseconds(channel, (uint16_t)us);
}

// Parses lineBuf into 16 floats and drives the servos.
// Returns true on success. Nothing is moved unless ALL 16 values parse, so a
// garbled line can't leave the hand half-updated.
bool handleLine(char *line) {
  float angles[NUM_CHANNELS];
  uint8_t count = 0;

  char *p = line;
  while (count < NUM_CHANNELS) {
    char *end;
    float v = strtod(p, &end);
    if (end == p) {                 // no number found
      Serial.println(F("ERR too few values"));
      return false;
    }
    angles[count++] = v;
    p = end;
    if (*p == ',') {
      p++;
    } else {
      break;                        // end of line or unexpected char
    }
  }

  while (*p == ' ' || *p == '\r') p++;   // tolerate trailing whitespace
  if (count != NUM_CHANNELS || *p != '\0') {
    Serial.println(F("ERR expected 16 values"));
    return false;
  }

  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    setServoAngle(i, angles[i]);
  }
  Serial.println(F("OK"));
  return true;
}

// ---- Arduino entry points --------------------------------------------------
void setup() {
  Serial.begin(BAUD_RATE);
  Wire.begin();
  Wire.setClock(400000);            // fast-mode I2C keeps 16-channel updates short

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);  // typical real PCA9685 oscillator
  pwm.setPWMFreq(PWM_FREQ_HZ);
  delay(10);

  // Start every servo at center (90 deg) so the hand comes up neutral.
  // Staggered one channel at a time so 16 servos don't all draw their
  // start-up surge at once (that is a common cause of brownouts/resets).
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    setServoAngle(i, 90.0);
    delay(40);
  }
}

void loop() {
  // Read bytes as they arrive instead of blocking on a whole line. The
  // Arduino's serial receive buffer is only 64 bytes, but a pose line is
  // ~97, so the loop must keep draining it or characters get dropped.
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      if (overflow) {
        Serial.println(F("ERR line too long"));
      } else {
        lineBuf[lineLen] = '\0';
        if (lineLen > 0) handleLine(lineBuf);
      }
      lineLen  = 0;
      overflow = false;
    } else if (lineLen < LINE_BUF_SIZE - 1) {
      lineBuf[lineLen++] = c;
    } else {
      overflow = true;              // discard until the next newline
    }
  }
}