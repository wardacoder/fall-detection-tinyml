/*
 * combined_benchmark.ino
 * - Measures gate (2 min) then CNN (2 min) at REALISTIC duty cycle
 * - One inference every 250 ms (matches deployed fall detector timing)
 * - INA260 in 128x averaging mode
 * - Progress and results reported via Serial Monitor (115200 baud)
 * - Tracks both inference-only AND end-to-end latency (sensor read +
 *   preprocess + inference + output read) for each model
 *
 * Wiring:
 *   INA260 : SDA=A4, SCL=A5, VCC=3V3, GND=GND, ADDR=GND -> 0x40
 *
 * Library: Arduino_TensorFlowLite (or Harvard_TinyMLx)
 */

#include <Wire.h>
#include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "gate_model_final.h"
#include "final_cnn_finetuned.h"

// ---------- INA260 ----------
#define INA260_ADDR         0x40
#define INA260_REG_CONFIG   0x00
#define INA260_REG_CURRENT  0x01

static const uint16_t INA260_CONFIG_AVG128 = 0x6927;

static void ina260_configure() {
  Wire.beginTransmission(INA260_ADDR);
  Wire.write(INA260_REG_CONFIG);
  Wire.write((INA260_CONFIG_AVG128 >> 8) & 0xFF);
  Wire.write(INA260_CONFIG_AVG128 & 0xFF);
  Wire.endTransmission();
}

static float readCurrent_mA() {
  Wire.beginTransmission(INA260_ADDR);
  Wire.write(INA260_REG_CURRENT);
  Wire.endTransmission(false);
  Wire.requestFrom(INA260_ADDR, 2);
  int16_t raw = (Wire.read() << 8) | Wire.read();
  return fabsf(raw * 1.25f);
}

// ---------- Window ----------
static const uint16_t WIN_LEN = 50;
static const uint8_t  N_CHAN  = 6;
static float win_ax[WIN_LEN], win_ay[WIN_LEN], win_az[WIN_LEN];
static float win_gx[WIN_LEN], win_gy[WIN_LEN], win_gz[WIN_LEN];

// ---------- TFLite ----------
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kArenaSize = 16 * 1024;
  alignas(16) uint8_t tensor_arena[kArenaSize];
}

// ---------- Run config ----------
static const uint32_t RUN_MS          = 120000UL; // 2 min per phase
static const uint16_t CUR_PERIOD_MS   = 2;        // 500 Hz INA260 sampling
static const uint16_t INFER_PERIOD_MS = 250;      // realistic: 1 inference / 250 ms
static const uint16_t PRINT_PERIOD_MS = 5000;     // Serial progress every 5 s

// ---------- Welford ----------
struct Welford {
  uint32_t n = 0;
  double mean = 0, M2 = 0;
  void add(double x) {
    n++;
    double d = x - mean;
    mean += d / n;
    M2 += d * (x - mean);
  }
  double stdev() const { return n > 1 ? sqrt(M2 / (n - 1)) : 0; }
};

static Welford gate_w, cnn_w;
static uint32_t gate_infer = 0, cnn_infer = 0;

// Inference-only latency (preprocess + model forward pass only)
static uint64_t gate_lat_sum = 0, cnn_lat_sum = 0;
static uint32_t gate_lat_min = 0xFFFFFFFFUL, gate_lat_max = 0;
static uint32_t cnn_lat_min  = 0xFFFFFFFFUL, cnn_lat_max  = 0;

// End-to-end latency (sensor read + preprocess + inference + output read)
static uint64_t gate_e2e_sum = 0, cnn_e2e_sum = 0;
static uint32_t gate_e2e_min = 0xFFFFFFFFUL, gate_e2e_max = 0;
static uint32_t cnn_e2e_min  = 0xFFFFFFFFUL, cnn_e2e_max  = 0;

// ---------- Serial progress ----------
static void serial_progress(const char* phase, uint32_t elapsed, uint32_t total, float live_mA, uint32_t inferences) {
  uint8_t pct = (uint8_t)((float)elapsed / total * 100.0f);
  Serial.print("["); Serial.print(phase); Serial.print("] ");
  Serial.print(elapsed / 1000); Serial.print("/"); Serial.print(total / 1000); Serial.print("s  ");
  Serial.print(pct); Serial.print("%  |  ");
  Serial.print("I="); Serial.print(live_mA, 2); Serial.print(" mA  |  ");
  Serial.print("inferences="); Serial.println(inferences);
}

static void serial_results() {
  uint32_t gate_lat_avg = gate_infer ? (uint32_t)(gate_lat_sum / gate_infer) : 0;
  uint32_t cnn_lat_avg  = cnn_infer  ? (uint32_t)(cnn_lat_sum  / cnn_infer)  : 0;
  uint32_t gate_e2e_avg = gate_infer ? (uint32_t)(gate_e2e_sum / gate_infer) : 0;
  uint32_t cnn_e2e_avg  = cnn_infer  ? (uint32_t)(cnn_e2e_sum  / cnn_infer)  : 0;

  Serial.println();
  Serial.println("========================================");
  Serial.println("           BENCHMARK RESULTS            ");
  Serial.println("========================================");

  Serial.println("--- Gate (logistic) ---");
  Serial.print("  Current  mean       : "); Serial.print(gate_w.mean,   3); Serial.println(" mA");
  Serial.print("  Current  stdev      : "); Serial.print(gate_w.stdev(),3); Serial.println(" mA");
  Serial.print("  Samples  (n)        : "); Serial.println(gate_w.n);
  Serial.print("  Inferences          : "); Serial.println(gate_infer);
  Serial.println("  -- Inference only (preprocess + forward pass) --");
  Serial.print("  Latency  min        : "); Serial.print(gate_lat_min); Serial.println(" us");
  Serial.print("  Latency  avg        : "); Serial.print(gate_lat_avg); Serial.println(" us");
  Serial.print("  Latency  max        : "); Serial.print(gate_lat_max); Serial.println(" us");
  Serial.println("  -- End-to-end (sensor read + preprocess + inference + output) --");
  Serial.print("  E2E Latency  min    : "); Serial.print(gate_e2e_min); Serial.println(" us");
  Serial.print("  E2E Latency  avg    : "); Serial.print(gate_e2e_avg); Serial.println(" us");
  Serial.print("  E2E Latency  max    : "); Serial.print(gate_e2e_max); Serial.println(" us");

  Serial.println();
  Serial.println("--- CNN (TFLite Micro) ---");
  Serial.print("  Current  mean       : "); Serial.print(cnn_w.mean,   3); Serial.println(" mA");
  Serial.print("  Current  stdev      : "); Serial.print(cnn_w.stdev(),3); Serial.println(" mA");
  Serial.print("  Samples  (n)        : "); Serial.println(cnn_w.n);
  Serial.print("  Inferences          : "); Serial.println(cnn_infer);
  Serial.println("  -- Inference only (copy input + Invoke) --");
  Serial.print("  Latency  min        : "); Serial.print(cnn_lat_min); Serial.println(" us");
  Serial.print("  Latency  avg        : "); Serial.print(cnn_lat_avg); Serial.println(" us");
  Serial.print("  Latency  max        : "); Serial.print(cnn_lat_max); Serial.println(" us");
  Serial.println("  -- End-to-end (sensor read + copy input + Invoke + output read) --");
  Serial.print("  E2E Latency  min    : "); Serial.print(cnn_e2e_min); Serial.println(" us");
  Serial.print("  E2E Latency  avg    : "); Serial.print(cnn_e2e_avg); Serial.println(" us");
  Serial.print("  E2E Latency  max    : "); Serial.print(cnn_e2e_max); Serial.println(" us");

  Serial.println("========================================");
}

// ---------- Gate ----------
static float run_gate(float gyro_sma, float ay_max, float svm_max) {
  float feats[3] = { gyro_sma, ay_max, svm_max };
  float z = GATE_INTERCEPT;
  for (uint8_t i = 0; i < 3; i++) {
    z += GATE_COEF[i] * ((feats[i] - GATE_MEANS[i]) / GATE_STDS[i]);
  }
  return 1.0f / (1.0f + expf(-z));
}

static void compute_features(float& gs, float& aym, float& svmm) {
  double g = 0; aym = 0; svmm = 0;
  for (uint16_t i = 0; i < WIN_LEN; i++) {
    g += fabsf(win_gx[i]) + fabsf(win_gy[i]) + fabsf(win_gz[i]);
    float a = fabsf(win_ay[i]); if (a > aym) aym = a;
    float s = sqrtf(win_ax[i]*win_ax[i] + win_ay[i]*win_ay[i] + win_az[i]*win_az[i]);
    if (s > svmm) svmm = s;
  }
  gs = g / WIN_LEN;
}

static void prefill_window() {
  uint16_t i = 0;
  while (i < WIN_LEN) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(win_ax[i], win_ay[i], win_az[i]);
      IMU.readGyroscope(win_gx[i], win_gy[i], win_gz[i]);
      i++;
    }
  }
}

static void copy_window_to_input() {
  float* in = input->data.f;
  for (uint16_t t = 0; t < WIN_LEN; t++) {
    in[t * N_CHAN + 0] = win_ax[t];
    in[t * N_CHAN + 1] = win_ay[t];
    in[t * N_CHAN + 2] = win_az[t];
    in[t * N_CHAN + 3] = win_gx[t];
    in[t * N_CHAN + 4] = win_gy[t];
    in[t * N_CHAN + 5] = win_gz[t];
  }
}

// ---------- Phase runners ----------
static void run_gate_phase() {
  Serial.println();
  Serial.println(">>> GATE PHASE starting (2 min, 1 inference / 250 ms)");
  uint32_t t_start    = millis();
  uint32_t next_cur   = t_start;
  uint32_t next_infer = t_start;
  uint32_t next_print = t_start;
  float last_mA = 0;

  while (millis() - t_start < RUN_MS) {
    // Sample current at 500 Hz
    if ((int32_t)(millis() - next_cur) >= 0) {
      last_mA = readCurrent_mA();
      gate_w.add(last_mA);
      next_cur += CUR_PERIOD_MS;
    }

    // Run gate at deployment cadence (every 250 ms)
    if ((int32_t)(millis() - next_infer) >= 0) {
      // --- E2E start: sensor read ---
      uint32_t e2e_ts = micros();
      float ax, ay, az, gx, gy, gz;
      while (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {}
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      // Shift window and insert new sample
      memmove(win_ax, win_ax + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_ay, win_ay + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_az, win_az + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_gx, win_gx + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_gy, win_gy + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_gz, win_gz + 1, (WIN_LEN - 1) * sizeof(float));
      win_ax[WIN_LEN-1] = ax; win_ay[WIN_LEN-1] = ay; win_az[WIN_LEN-1] = az;
      win_gx[WIN_LEN-1] = gx; win_gy[WIN_LEN-1] = gy; win_gz[WIN_LEN-1] = gz;

      // --- preprocess + inference ---
      float gs, aym, svmm;
      compute_features(gs, aym, svmm);
      uint32_t infer_ts = micros();
      float prob = run_gate(gs, aym, svmm);
      uint32_t infer_dt = micros() - infer_ts;

      // --- output read (threshold check = "output") ---
      bool triggered = (prob >= GATE_THRESHOLD);
      (void)triggered;
      uint32_t e2e_dt = micros() - e2e_ts;

      gate_infer++;
      gate_lat_sum += infer_dt;
      if (infer_dt < gate_lat_min) gate_lat_min = infer_dt;
      if (infer_dt > gate_lat_max) gate_lat_max = infer_dt;
      gate_e2e_sum += e2e_dt;
      if (e2e_dt < gate_e2e_min) gate_e2e_min = e2e_dt;
      if (e2e_dt > gate_e2e_max) gate_e2e_max = e2e_dt;
      next_infer += INFER_PERIOD_MS;
    }

    // Print progress every 5 s
    if ((int32_t)(millis() - next_print) >= 0) {
      serial_progress("GATE", millis() - t_start, RUN_MS, last_mA, gate_infer);
      next_print += PRINT_PERIOD_MS;
    }
  }
  Serial.println(">>> GATE PHASE done");
}

static void run_cnn_phase() {
  Serial.println();
  Serial.println(">>> CNN PHASE starting (2 min, 1 inference / 250 ms)");
  uint32_t t_start    = millis();
  uint32_t next_cur   = t_start;
  uint32_t next_infer = t_start;
  uint32_t next_print = t_start;
  float last_mA = 0;

  while (millis() - t_start < RUN_MS) {
    if ((int32_t)(millis() - next_cur) >= 0) {
      last_mA = readCurrent_mA();
      cnn_w.add(last_mA);
      next_cur += CUR_PERIOD_MS;
    }

    if ((int32_t)(millis() - next_infer) >= 0) {
      // --- E2E start: sensor read ---
      uint32_t e2e_ts = micros();
      float ax, ay, az, gx, gy, gz;
      while (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {}
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      // Shift window and insert new sample
      memmove(win_ax, win_ax + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_ay, win_ay + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_az, win_az + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_gx, win_gx + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_gy, win_gy + 1, (WIN_LEN - 1) * sizeof(float));
      memmove(win_gz, win_gz + 1, (WIN_LEN - 1) * sizeof(float));
      win_ax[WIN_LEN-1] = ax; win_ay[WIN_LEN-1] = ay; win_az[WIN_LEN-1] = az;
      win_gx[WIN_LEN-1] = gx; win_gy[WIN_LEN-1] = gy; win_gz[WIN_LEN-1] = gz;

      // --- copy input + inference ---
      copy_window_to_input();
      uint32_t infer_ts = micros();
      TfLiteStatus s = interpreter->Invoke();
      uint32_t infer_dt = micros() - infer_ts;

      // --- output read ---
      float cnn_out = output->data.f[0];
      (void)cnn_out;
      uint32_t e2e_dt = micros() - e2e_ts;

      if (s == kTfLiteOk) {
        cnn_infer++;
        cnn_lat_sum += infer_dt;
        if (infer_dt < cnn_lat_min) cnn_lat_min = infer_dt;
        if (infer_dt > cnn_lat_max) cnn_lat_max = infer_dt;
        cnn_e2e_sum += e2e_dt;
        if (e2e_dt < cnn_e2e_min) cnn_e2e_min = e2e_dt;
        if (e2e_dt > cnn_e2e_max) cnn_e2e_max = e2e_dt;
      }
      next_infer += INFER_PERIOD_MS;
    }

    if ((int32_t)(millis() - next_print) >= 0) {
      serial_progress("CNN ", millis() - t_start, RUN_MS, last_mA, cnn_infer);
      next_print += PRINT_PERIOD_MS;
    }
  }
  Serial.println(">>> CNN PHASE done");
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}

  Serial.println("========================================");
  Serial.println("       Combined Benchmark Booting       ");
  Serial.println("========================================");

  Wire.begin();
  Wire.setClock(400000);

  if (!IMU.begin()) { Serial.println("ERROR: IMU init failed"); while(1){} }
  Serial.println("IMU    ... OK");

  ina260_configure();
  Serial.println("INA260 ... OK");
  delay(50);

  // ---- TFLite Micro init ----
  model = tflite::GetModel(final_cnn_finetuned_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: TFLite schema mismatch"); while(1){}
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interp(model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interp;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed"); while(1){}
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("TFLite ... OK");

  prefill_window();
  Serial.println("Window prefilled. Starting benchmark...");
  delay(500);

  run_gate_phase();
  run_cnn_phase();

  serial_results();
}

void loop() {}