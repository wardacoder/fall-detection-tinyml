# Selective Three-Stage TinyML Inference for Real-Time Fall Detection

On-device fall detection running entirely on an **Arduino Nano 33 BLE Sense Rev1**, with no phone, no cloud, and no connectivity required. The system was motivated by a simple real-world problem: elderly people who live alone often fall with no one nearby to respond, and most existing detectors depend on a network connection or a charged phone. This one runs standalone on a coin-sized board.

The core idea is that falls are rare events. Running a neural network on every window of sensor data wastes energy during the long stretches of ordinary daily activity that dominate any real stream. This system uses a selective cascade that resolves the overwhelming majority of activity with cheap arithmetic and only wakes the neural network when an event actually looks like a fall. The result matches always-on accuracy while cutting per-event energy by **98.4%**.

| Metric | Result |
|---|---|
| Live accuracy (40 trials) | **97.5%** (39/40) |
| Fall F1 (live) | **0.974** |
| Energy per event vs always-on CNN | **8.97 uJ vs 565.95 uJ (98.4% lower)** |
| Deployed model size | **5.86 KB** CNN + 40 B gate |
| Gate latency | **35 us** |
| CNN latency | **5.82 ms** |
| RAM footprint | 86 KB of 256 KB (33%) |

---

## How it works

Most TinyML fall detectors run a deep model on every fixed window, regardless of whether anything is happening. That burns energy during the long periods of ordinary daily activity that dominate any real sensor stream. This system inverts that with a three-stage cascade, where each stage is more expensive than the last and only runs if the previous stage passes.

```
IMU stream (LSM9DS1 @ 100 Hz)
        |
        v
+-----------------------------------------------+
|  Stage 0 . SVM impact pre-filter              |   arithmetic, no model
|  reject windows below 1.5g                    |   resolves most daily activity here
+-----------------------------------------------+
        | impact + post-impact stillness
        v
+-----------------------------------------------+
|  Stage 1 . Logistic-regression gate           |   35 us, 3 handcrafted features
|  gyro SMA, vertical accel peak, SVM peak      |   plain C, no interpreter
|  pass if probability >= 0.26                  |
+-----------------------------------------------+
        | gate pass
        v
+-----------------------------------------------+
|  Stage 2 . CNN verifier                       |   5.82 ms, raw [50 x 6] window
|  fires only on gate-flagged windows           |   float32 TFLite, 5.86 KB
|  report fall if probability >= 0.90           |
+-----------------------------------------------+
        |
        v
   Fall / No Fall  (buzzer, OLED, Serial)
```

A pure walking sequence never reaches a model at all, since it is rejected arithmetically at Stage 0. The CNN, the most expensive component, runs only on the rare windows that survive both earlier gates. That selectivity is the entire reason the energy number is what it is.

---

## Results across development

The system was built in stages, each one targeting the weakness of the last. Live accuracy climbed from 85% to 97.5% while per-event energy dropped by two orders of magnitude.

| Version | Live accuracy | Live F1 | Energy/event | Latency |
|---|---|---|---|---|
| Baseline (CNN only, always on) | 85.0% | 0.857 | 565.95 uJ | 5.96 ms |
| + Gate and fine-tuned CNN | 95.0% | 0.947 | 8.97 uJ | 0.035 ms |
| + Impact pre-filter (final) | **97.5%** | **0.974** | **8.97 uJ** | **0.035 ms** |

The latency drop comes from the gate resolving the dominant path in 35 microseconds, so the CNN's 5.82 ms cost is only paid on the rare windows that reach it.

---

## Engineering decisions worth highlighting

**Float32 over quantization.** The deployed CNN is not quantized. At 5.86 KB the float32 export is jointly the smallest and most accurate variant: FP16 grew to 6.12 KB from per-tensor metadata, and full int8 dropped F1 from 0.967 to 0.842. The Cortex-M4F hardware FPU makes the float32 latency cost negligible. Quantization was tested and rejected on evidence, not assumed.

**Gate exported as plain C, not a model.** The always-on Stage 1 gate is a logistic regression compiled to four constant arrays and a sigmoid in C, with no TFLite interpreter overhead on the path that runs continuously. That is where the 35 microsecond figure comes from.

**Domain-gap-aware training.** Trained on the public SisFall dataset, then gate-retrained and CNN-fine-tuned on a separately collected Arduino dataset, because the deployment IMU (LSM9DS1) differs from the dataset sensor. Thresholds were derived from train and validation data only, with the test set locked before any pipeline design.

---

## Repository structure

```
.
|-- README.md
|-- requirements.txt
|-- arduino/
|   |-- deployment/           Deployment sketch + model headers
|   `-- power_benchmark/      Power benchmarking sketch (INA260, 481 inferences)
|-- notebooks/                00 to 07, full pipeline from raw data to deployable model
|-- models/                   Keras source, deployed .tflite, gate C header, thresholds
|-- data/
|   |-- arduino/              Sample collected CSVs (format reference)
|   |-- arduino_preprocessed/ Windowed and labelled test split
|   `-- README.md             SisFall download instructions
`-- results/
    |-- confusion_matrices/   Offline and live
    |-- progression_table.csv Metric progression across development stages
    `-- quantization_tradeoff.png
```

---

## Reproducing

**Python** (tested on 3.10 and 3.12):

```bash
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run notebooks `00` through `07` in order. Each has a config cell at the top, so update the paths before running. End-to-end runtime on Colab with a GPU is roughly 15 to 20 minutes. SisFall is not committed to the repo. See [`data/README.md`](data/README.md) for download instructions.

**Arduino** (IDE 2.x or arduino-cli):

1. Board package: Arduino Mbed OS Nano Boards v4.1.5, then select Arduino Nano 33 BLE.
2. Libraries: `Arduino_LSM9DS1` 1.1.1 and `Arduino_TensorFlowLite` 2.4.0-ALPHA, plus `Adafruit_INA260` and `Adafruit_SSD1306` for the power bench and OLED.
3. Open `arduino/deployment/deployment.ino`, flash, and open the Serial Monitor at 115200 baud.

The model headers are already in the sketch folder. The conversion path from Keras to TFLite to C array lives in notebook `07`.

---

## License

Licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Attribution required, no commercial use, and no derivative works.
