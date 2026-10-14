/*
 * Project SPANDAN: Edge Sentry Node Firmware
 * Hardware: ESP32-WROOM-32 + MPU-6050 (I2C)
 * Features:
 *   - Deterministic 16-bit I2C reads (fixes compiler byte reordering)
 *   - Non-blocking serial command listener (prevents sample drops)
 *   - Jitter-free 100.0 Hz timer accumulation
 *   - Automatic stationary gravity bias removal at boot
 */

#include <Wire.h>

#define MPU_ADDR 0x68
#define ALERT_PIN 2               // Built-in LED or external buzzer/relay
#define SAMPLING_FREQ_HZ 100
#define SAMPLE_INTERVAL_US (1000000 / SAMPLING_FREQ_HZ)

float az_offset = 0.0;
unsigned long startTimeUs = 0;
unsigned long lastSampleUs = 0;

// Deterministic 16-bit I2C reading (Fixes GCC unspecified evaluation order bug)
int16_t readRawAxis(uint8_t regAddr) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(regAddr);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)2, (uint8_t)true);
  
  if (Wire.available() >= 2) {
    uint8_t msb = Wire.read();
    uint8_t lsb = Wire.read();
    return (int16_t)((msb << 8) | lsb);
  }
  return 0;
}

void calibrateStationaryBias() {
  const int calibSamples = 400;
  long raw_z_sum = 0;
  for (int i = 0; i < calibSamples; i++) {
    raw_z_sum += readRawAxis(0x3F); // ACCEL_ZOUT_H
    delay(2);
  }
  // 16384 LSB/g at +/- 2g range -> Convert to m/s^2
  az_offset = ((float)raw_z_sum / (float)calibSamples) / 16384.0f * 9.80665f;
}

// Non-blocking serial command listener (Replaces blocking readStringUntil)
void checkIncomingCommandsNonBlocking() {
  static String inputBuffer = "";
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      inputBuffer.trim();
      if (inputBuffer == "CMD:ALARM_ON") {
        digitalWrite(ALERT_PIN, HIGH);
      } else if (inputBuffer == "CMD:ALARM_OFF") {
        digitalWrite(ALERT_PIN, LOW);
      }
      inputBuffer = "";
    } else {
      inputBuffer += inChar;
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ALERT_PIN, OUTPUT);
  digitalWrite(ALERT_PIN, LOW);

  Wire.begin(21, 22);             // SDA = GPIO 21, SCL = GPIO 22
  Wire.setClock(400000);          // Fast I2C (400 kHz)

  // Wake up MPU-6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);               // PWR_MGMT_1
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Set Accelerometer Full-Scale Range to +/- 2g (16384 LSB/g)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);               // ACCEL_CONFIG
  Wire.write(0x00);               // 0x00 = +/- 2g
  Wire.endTransmission(true);

  calibrateStationaryBias();
  startTimeUs = micros();
  lastSampleUs = startTimeUs;
}

void loop() {
  checkIncomingCommandsNonBlocking();

  unsigned long currentUs = micros();
  // Fixed interval accumulation to eliminate sampling phase drift
  if (currentUs - lastSampleUs >= SAMPLE_INTERVAL_US) {
    lastSampleUs += SAMPLE_INTERVAL_US;

    float t_sec = (currentUs - startTimeUs) / 1000000.0f;
    int16_t raw_z = readRawAxis(0x3F);
    float a_dynamic = ((float)raw_z / 16384.0f * 9.80665f) - az_offset;

    // Stream telemetry: timestamp_seconds, dynamic_accel_m_s2
    Serial.print(t_sec, 4);
    Serial.print(",");
    Serial.println(a_dynamic, 4);
  }
}
