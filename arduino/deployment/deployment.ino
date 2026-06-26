/*
 * Fall Detector — Live Three-Stage Pipeline
 * Arduino Nano 33 BLE Sense Rev1
 */

#include <Wire.h>
#include <ACROBOTIC_SSD1306.h>
#include <Arduino_LSM9DS1.h>
#include <MicroTFLite.h>
#include "gate_model_final.h"
#include "final_cnn_finetuned.h"

// ── Window parameters ─────────────────────────────────────────────────
#define WIN_SAMPLES   50
#define N_CHANNELS    6
#define FALL_PRE      25
#define FALL_POST     25

// ── Impact detection thresholds ───────────────────────────────────────
#define IMPACT_THRESH   1.5f
#define STILL_MEAN      1.20f
#define MIN_STILL       50

// ── Ring buffer ───────────────────────────────────────────────────────
#define RING_SIZE  600
float ring_ax[RING_SIZE], ring_ay[RING_SIZE], ring_az[RING_SIZE];
float ring_gx[RING_SIZE], ring_gy[RING_SIZE], ring_gz[RING_SIZE];
float ring_svm[RING_SIZE];
int   ring_head  = 0;
int   ring_count = 0;

// ── Window buffer — GLOBAL to avoid stack overflow ────────────────────
float g_window[WIN_SAMPLES][N_CHANNELS];

// ── State machine ─────────────────────────────────────────────────────
enum State { MONITORING, WAITING_STILLNESS, PROCESSING };
State state = MONITORING;

int   impact_ring_idx = -1;
float impact_svm_peak = 0.0f;
int   still_start_idx = -1;
int   still_count     = 0;

// ── Thresholds ────────────────────────────────────────────────────────
#define CNN_THRESHOLD  0.90f

// ── Buzzer ────────────────────────────────────────────────────────────
#define BUZZER_PIN  9

// ── INA260 ────────────────────────────────────────────────────────────
#define INA260_ADDR         0x40
#define INA260_REG_CURRENT  0x01

// ── TFLite ────────────────────────────────────────────────────────────
constexpr int kTensorArenaSize = 12 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

// ── Calibration ───────────────────────────────────────────────────────
#define CAL_AX   0.01842f
#define CAL_AY  -0.01293f
#define CAL_AZ  -0.08174f
#define CAL_GX   9.516f
#define CAL_GY   1.098f
#define CAL_GZ   1.098f

// ── Stats ─────────────────────────────────────────────────────────────
unsigned long impacts_detected = 0;
unsigned long gate_passed      = 0;
unsigned long falls_detected   = 0;
unsigned long last_fall_ms     = 0;
#define FALL_COOLDOWN_MS  3000

// ── INA260 ────────────────────────────────────────────────────────────
float readCurrent_mA() {
  Wire.beginTransmission(INA260_ADDR);
  Wire.write(INA260_REG_CURRENT);
  Wire.endTransmission(false);
  Wire.requestFrom(INA260_ADDR, 2);
  int16_t raw = (Wire.read() << 8) | Wire.read();
  return abs(raw * 1.25f);
}

// ── Buzzer ────────────────────────────────────────────────────────────
void buzzFallAlert() {
  tone(BUZZER_PIN, 800,  150); delay(200);
  tone(BUZZER_PIN, 1200, 150); delay(200);
  tone(BUZZER_PIN, 1600, 400);
}

void buzzStartup() {
  tone(BUZZER_PIN, 1000, 150); delay(200);
  tone(BUZZER_PIN, 1500, 150); delay(200);
  tone(BUZZER_PIN, 2000, 200); delay(300);
  noTone(BUZZER_PIN);
}

void buzzImpact() {
  tone(BUZZER_PIN, 600, 80);
}

// ── OLED ──────────────────────────────────────────────────────────────
void oledLine(uint8_t row, String msg) {
  oled.setTextXY(row, 0);
  oled.putString(msg.substring(0, 16));
}

void oledShowMonitoring() {
  oled.clearDisplay();
  oledLine(0, "Fall Detector");
  oledLine(1, "Monitoring...");
  oledLine(3, "Impacts:" + String(impacts_detected));
  oledLine(4, "Falls:  " + String(falls_detected));
}

void oledShowWaiting() {
  oled.clearDisplay();
  oledLine(0, "Impact detected!");
  oledLine(2, "Waiting for");
  oledLine(3, "stillness...");
  oledLine(5, "Peak:" + String(impact_svm_peak, 2) + "g");
}

void oledShowFall(float gate_prob, float cnn_prob, float current_mA) {
  oled.clearDisplay();
  oledLine(0, "!!! FALL !!!");
  oledLine(1, "Gate:" + String(gate_prob, 3));
  oledLine(2, "CNN: " + String(cnn_prob,  3));
  oledLine(3, "mA:  " + String(current_mA, 1));
  oledLine(5, "N:   " + String(falls_detected));
}

void oledShowADL(float gate_prob) {
  oled.clearDisplay();
  oledLine(0, "Impact -> ADL");
  oledLine(2, "Gate:" + String(gate_prob, 3));
  oledLine(4, "Resuming...");
}

// ── Ring buffer helpers ───────────────────────────────────────────────
int ringIdx(int abs_idx) {
  return abs_idx % RING_SIZE;
}

bool extractWindow(int peak_abs_idx) {
  int start = peak_abs_idx - FALL_PRE;
  if (start < 0) return false;
  if ((ring_count - start) > RING_SIZE) return false;

  for (int i = 0; i < WIN_SAMPLES; i++) {
    int idx = ringIdx(start + i);
    g_window[i][0] = ring_ax[idx];
    g_window[i][1] = ring_ay[idx];
    g_window[i][2] = ring_az[idx];
    g_window[i][3] = ring_gx[idx];
    g_window[i][4] = ring_gy[idx];
    g_window[i][5] = ring_gz[idx];
  }
  return true;
}

// ── Gate ──────────────────────────────────────────────────────────────
float gatePredict() {
  float ay_max   = -9999.0f;
  float svm_max  = 0.0f;
  float gyro_sma = 0.0f;

  for (int t = 0; t < WIN_SAMPLES; t++) {
    float ax = g_window[t][0], ay = g_window[t][1], az = g_window[t][2];
    float gx = g_window[t][3], gy = g_window[t][4], gz = g_window[t][5];
    if (ay > ay_max) ay_max = ay;
    float svm = sqrt(ax*ax + ay*ay + az*az);
    if (svm > svm_max) svm_max = svm;
    gyro_sma += abs(gx) + abs(gy) + abs(gz);
  }
  gyro_sma /= WIN_SAMPLES;

  float features[GATE_N_ACTIVE] = {gyro_sma, ay_max, svm_max};
  float logit = GATE_INTERCEPT;
  for (int i = 0; i < GATE_N_ACTIVE; i++) {
    float scaled = (features[i] - GATE_MEANS[i]) / GATE_STDS[i];
    logit += GATE_COEF[i] * scaled;
  }
  return 1.0f / (1.0f + exp(-logit));
}

// ── CNN ───────────────────────────────────────────────────────────────
float cnnPredict() {
  // Flatten window into input buffer
  float input_buf[WIN_SAMPLES * N_CHANNELS];
  for (int t = 0; t < WIN_SAMPLES; t++) {
    for (int c = 0; c < N_CHANNELS; c++) {
      input_buf[t * N_CHANNELS + c] = g_window[t][c];
    }
  }

  // Feed all 300 values — swapped argument order (index, value)
  for (int i = 0; i < WIN_SAMPLES * N_CHANNELS; i++) {
    if (!ModelSetInput(input_buf[i], i)) {
      Serial.print("[CNN ERROR] SetInput failed at i=");
      Serial.println(i);
      return -1.0f;
    }
  }

  Serial.println("[CNN] Running inference...");
  bool success = ModelRunInference();
  Serial.print("[CNN] Inference done, success="); Serial.println(success);

  if (!success) {
    Serial.println("[CNN ERROR] Inference failed");
    return -1.0f;
  }

  float result = ModelGetOutput(0);
  Serial.print("[CNN] Raw output="); Serial.println(result, 6);
  return result;
}

// ── Setup ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  pinMode(BUZZER_PIN, OUTPUT);
  noTone(BUZZER_PIN);
  digitalWrite(BUZZER_PIN, LOW);

  oled.init();
  oled.clearDisplay();
  oledLine(0, "Fall Detector");
  oledLine(2, "Initializing...");

  if (!IMU.begin()) {
    oledLine(4, "IMU FAIL");
    Serial.println("[ERROR] IMU init failed");
    while (1);
  }

  if (!ModelInit(final_cnn_finetuned_tflite,
                 tensorArena, kTensorArenaSize)) {
    oledLine(4, "CNN FAIL");
    Serial.println("[ERROR] TFLite model init failed");
    while (1);
  }

  buzzStartup();

  Serial.println("=== FALL DETECTOR STARTED ===");
  Serial.println("Pipeline: impact → gate → CNN");
  Serial.print("Impact threshold : "); Serial.print(IMPACT_THRESH);  Serial.println("g");
  Serial.print("Gate threshold   : "); Serial.println(GATE_THRESHOLD);
  Serial.print("CNN  threshold   : "); Serial.println(CNN_THRESHOLD);
  Serial.println("=============================================");
  Serial.println("Monitoring for falls...");

  oled.clearDisplay();
  oledLine(0, "Fall Detector");
  oledLine(2, "Ready!");
  oledLine(4, "Monitoring...");
  delay(1000);
  oledShowMonitoring();
}

// ── Loop ──────────────────────────────────────────────────────────────
void loop() {
  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) return;

  float ax, ay, az, gx, gy, gz;
  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  ax -= CAL_AX; ay -= CAL_AY; az -= CAL_AZ;
  gx -= CAL_GX; gy -= CAL_GY; gz -= CAL_GZ;

  float svm = sqrt(ax*ax + ay*ay + az*az);

  // Store in ring buffer
  int slot      = ring_head;
  ring_ax[slot] = ax; ring_ay[slot] = ay; ring_az[slot] = az;
  ring_gx[slot] = gx; ring_gy[slot] = gy; ring_gz[slot] = gz;
  ring_svm[slot] = svm;
  ring_head  = (ring_head + 1) % RING_SIZE;
  ring_count++;

  switch (state) {

    case MONITORING: {
      if (svm >= IMPACT_THRESH) {
        impact_ring_idx = ring_count - 1;
        impact_svm_peak = svm;
        still_count     = 0;
        still_start_idx = -1;
        state           = WAITING_STILLNESS;
        impacts_detected++;
        buzzImpact();
        oledShowWaiting();
        Serial.print("[IMPACT] SVM="); Serial.print(svm, 3);
        Serial.print("g at sample "); Serial.println(ring_count);
      }
      break;
    }

    case WAITING_STILLNESS: {
      // Refine peak within 50 samples
      if (svm > impact_svm_peak &&
          (ring_count - 1 - impact_ring_idx) < 50) {
        impact_ring_idx = ring_count - 1;
        impact_svm_peak = svm;
      }

      // Stillness check
      if (svm < STILL_MEAN) {
        if (still_start_idx == -1) still_start_idx = ring_count - 1;
        still_count++;
      } else {
        still_count     = 0;
        still_start_idx = -1;
      }

      // Timeout after 5 seconds
      if ((ring_count - 1 - impact_ring_idx) > 500) {
        Serial.println("[TIMEOUT] No stillness — discarding");
        state = MONITORING;
        oledShowMonitoring();
        break;
      }

      // Stillness confirmed — move to processing
      if (still_count >= MIN_STILL) {
        state = PROCESSING;
      }
      break;
    }

    case PROCESSING: {
      Serial.println("[PROCESSING] Running pipeline...");

      bool ok = extractWindow(impact_ring_idx);
      if (!ok) {
        Serial.println("[ERROR] Window extraction failed");
        state = MONITORING;
        oledShowMonitoring();
        break;
      }

      float current_before  = readCurrent_mA();
      unsigned long t_start = micros();

      float gate_prob = gatePredict();
      float cnn_prob  = -1.0f;
      bool  fall      = false;

      Serial.print("[GATE] prob="); Serial.println(gate_prob, 4);

      if (gate_prob >= GATE_THRESHOLD) {
        gate_passed++;
        cnn_prob = cnnPredict();
        Serial.print("[CNN]  prob="); Serial.println(cnn_prob, 4);
        if (cnn_prob >= CNN_THRESHOLD) fall = true;
      }

      unsigned long latency_us = micros() - t_start;
      float current_after = readCurrent_mA();
      float current_mA    = (current_before + current_after) / 2.0f;

      if (fall) {
        unsigned long now_ms = millis();
        if (now_ms - last_fall_ms > FALL_COOLDOWN_MS) {
          last_fall_ms = now_ms;
          falls_detected++;
          Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
          Serial.println("!!! FALL DETECTED               !!!");
          Serial.print("  Gate    : "); Serial.println(gate_prob,  4);
          Serial.print("  CNN     : "); Serial.println(cnn_prob,   4);
          Serial.print("  Current : "); Serial.print(current_mA, 2); Serial.println(" mA");
          Serial.print("  Latency : "); Serial.print(latency_us/1000.0f, 2); Serial.println(" ms");
          Serial.print("  Total   : "); Serial.println(falls_detected);
          Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
          buzzFallAlert();
          oledShowFall(gate_prob, cnn_prob, current_mA);
          delay(3000);
        }
      } else {
        Serial.println("[RESULT] ADL — no fall");
        oledShowADL(gate_prob);
        delay(1000);
      }

      state = MONITORING;
      oledShowMonitoring();
      break;
    }
  }
}