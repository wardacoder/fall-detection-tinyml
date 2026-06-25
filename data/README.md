# Data

All raw and preprocessed datasets are hosted on Google Drive due to GitHub file-size restrictions. Nothing in this folder requires local copies to run the notebooks — each notebook cell that loads data points to the relevant subfolder path, so downloading to the expected location is sufficient.

---

## Folder Structure

```
data/
├── sisfall/                   SisFall raw sample trials
├── arduino/                   Arduino-collected CSVs per volunteer
│   ├── V2_Shams/
│   ├── V7_Warda/
│   └── ...
├── arduino_preprocessed/      Windowed + labelled Arduino dataset (all splits)
│   └── adl_test.csv           166 ADL windows × 50 timesteps
└── README.md                  ← you are here
```

---

## Downloads

> **None of the datasets are tracked by Git.** Download each folder from the links below and place them at the paths shown.

| Folder | Contents | Size note | Link |
|--------|----------|-----------|------|
| `data/sisfall/` | SisFall raw trial CSVs (MMA8451Q + ITG3200, 200 Hz, 38 subjects) | Too large for GitHub | [Download from Google Drive](https://drive.google.com/drive/folders/16iJZYV9khtKD0cz-2rMdkaVqtWaLVIIV?usp=sharing) |
| `data/sisfall_preprocessed/` | SisFall windowed + labelled splits | Too large for GitHub | [Download from Google Drive](https://drive.google.com/drive/folders/1yNh8iB8TclJG3f0e1Vv3f3QG6wkFbgZo?usp=sharing) |
| `data/arduino/` | Per-volunteer raw CSVs (LSM9DS1, 100 Hz) | Too large for GitHub | [Download from Google Drive](https://drive.google.com/drive/folders/19MDf1K9n3V8cCo_44cfHPKixvqFcTlU1?usp=sharing) |
| `data/arduino_preprocessed/` | Windowed + labelled Arduino dataset (all splits) | Too large for GitHub | [Download from Google Drive](https://drive.google.com/drive/folders/1EDMaZ5IlMbnDYfHTdmLv3czYp5n3_V2I?usp=sharing) |

---

## Dataset Details

### SisFall

| Field | Value |
|-------|-------|
| Source | [Sucerquia et al., *Sensors* 2017](https://doi.org/10.3390/s17010198) |
| Licence | CC BY 4.0 |
| Sensors | MMA8451Q accelerometer (±8 g) + ITG3200 gyroscope |
| Original rate | 200 Hz → decimated to 100 Hz for this project |
| Subjects | 38 (23 young adults, 15 elderly) |
| Activities | 15 fall types (F01–F15) + 19 ADL types (D01–D19) |
| Placement | Waist-mounted |
| Windows | 500 ms / 50 samples, 60 % overlap (200 ms stride) |
| Total windows | 16,037 (1,775 fall · 14,262 ADL) |
| Split | 70 / 15 / 15 — trial-level stratified, test set locked |

Only the MMA8451Q (±8 g) channel is retained; the ADXL345 (±16 g) is discarded. The ±8 g range more closely matches the LSM9DS1 deployment sensor and provides higher effective sensitivity in the 1–8 g fall-impact band.

---

### Arduino-Collected Dataset

| Field | Value |
|-------|-------|
| Licence | CC BY 4.0 |
| Sensor | LSM9DS1 (onboard Arduino Nano 33 BLE Sense Rev1) |
| Rate | 100 Hz (native) |
| Placement | Front waist, belly-button height (mirrors SisFall protocol) |
| Subjects | 11 total — 7 performed falls + ADLs, 4 performed ADLs only |
| Environments | 2 indoor home settings |
| Activities | F01–F15 fall types + D01–D19 ADL types (SisFall taxonomy) |
| Capture | CoolTerm CSV at 115,200 baud; trials outside ±0.5 Hz of 100 Hz discarded |
| Calibration | Per-axis offsets measured at session start, applied before processing |
| Role | Fine-tuning + live on-device evaluation only (not used for SisFall test-set metrics) |

Raw CSVs are organised by volunteer ID under `data/arduino/` (e.g. `V2_Shams/`, `V7_Warda/`).

---

### Arduino Preprocessed

Windowed and labelled version of the Arduino dataset, produced by the same pipeline used for SisFall (500 ms windows, 50 samples, trial-level 70/15/15 split).

| Split | Fall | ADL | Total |
|-------|------|-----|-------|
| Train | 77 | 125 | 202 |
| Val | 17 | 27 | 44 |
| Test | 17 | 27 | 44 |
| **Total** | **111** | **179** | **290** |

The test set was locked before any model selection or threshold tuning.

---

## Licence

- **SisFall** data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — cite Sucerquia et al. (2017) when using.  
- **Arduino-collected** data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
