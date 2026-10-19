# Wrist activity recognition + fall detection — SiWG917 prototype

Daily-activity recognition (trained on **PAMAP2**) and fall detection (trained on **BITS-2**), exported as small **INT8 TensorFlow Lite**
models for a wrist device built around the Silicon Labs **SiWG917**.

> **Status (phase 1):** models trained and validated on public data, exported to INT8 TFLite.
> **Not yet tested on a person wearing the device, and not yet running on the chip.** All numbers below are from public datasets
> and should be treated as provisional.

---

## 1. Goal
1. Recognise normal daily activities from a wrist-worn accelerometer + gyroscope.
2. Detect falls.
3. Use the BITS-2 wrist recordings to *reinforce* the PAMAP2 activity model (same activities should have similar accelerometer/gyro ranges).
4. Run both models on the SiWG917 (TFLite Micro, INT8).

## 2. Datasets
| Dataset | Used for | Notes |
|---|---|---|
| **PAMAP2** (UCI) | Activity training | Dominant-hand IMU, 100 Hz, acc in m/s² *with gravity*, gyro in rad/s |
| **BITS-2** (*Data in Brief*, 2024) | Fall detection | Custom wrist-worn device; streams `acc` (linear, gravity removed), `acg` (with gravity), `gyro`, `mgm`; 16 ADLs + falls; 41 users |
| **MHEALTH** (UCI) | Tested as extra wrist data, **not used for training** | Right lower arm, 50 Hz |

Datasets are **not** included in this repo. Download them yourself and follow their licences / citation requirements.
Expected paths: `/content/PAMAP2_Data/PAMAP2_Dataset/Protocol`, `/content/BITS-2/Dataset` (override with env vars `PAMAP_DIR`, `BITS_DIR`).

## 3. What we found along the way (progress log)

### 3.1 Why the first PAMAP2 ↔ BITS-2 comparison scored poorly
- BITS-2 `acc` is centred on 0 (gravity removed) while PAMAP2's accelerometer includes gravity → must use BITS-2 **`acg`**.
- Per-axis mean/min/max depend on watch orientation → use **magnitudes and sorted per-axis std** instead.
- Features such as mean|Δ| depend on sampling rate → dropped.
- BITS-2 timestamps are stored with 3 significant digits (`2.92E+12`), so the sampling rate cannot be recovered from them; windows are by sample count.
- `adl1…adl16` are numbered activities from the paper's Table 1, not class names.
- Sanity checks after the fix: quiet-window |acc| = 9.72 (PAMAP2) and 9.87 m/s² (BITS-2 `acg`) → units/gravity consistent. A walking-cadence estimate put the BITS-2 rate at up to ≈57 Hz (paper: 20 Hz) → **BITS-2 sampling rate remains uncertain**.

### 3.2 Does BITS-2 reinforce PAMAP2? — No (on unseen BITS-2 users, 4 shared dynamic classes)
| Model | Accuracy |
|---|---|
| PAMAP2 only (calibrated) | 0.468 |
| BITS-2 only | **0.684** |
| Joint training | 0.669 |
| Fusion (best) | 0.672 |

Only jogging transferred cleanly. BITS-2 can only help dynamic activities (walking, stairs, running), not static postures.

### 3.3 Does MHEALTH reinforce PAMAP2? — Not meaningfully
- MHEALTH's gyroscope columns turned out to be on a different/unreliable scale (flat 0.02 across all classes after conversion) → gyro excluded.
- Accelerometer-only, PAMAP2 → MHEALTH (zero MHEALTH training): 0.482 accuracy (running F1 0.76, cycling 0.71; lying/sitting/standing 0.22–0.36).
- MHEALTH-only 0.730 · joint 0.723 · best fusion 0.738 (within noise with 10 subjects).
- **Decision:** MHEALTH is not used for training. PAMAP2 trains the activity model.

### 3.4 Fall detection: label noise was the main problem
- v1 labelled *every* window of a fall recording as "fall" → at ≈85 % recall, 22 % of ADL recordings false-alarmed.
- v2 **impact-centred labels** (only windows containing the acceleration peak are positives; other windows of fall recordings are excluded from training; decision per recording) → Random Forest: recall 0.85, ADL false alarm 0.10, precision 0.81 (threshold 0.4).

### 3.5 Deployment design decisions
- **lying / sitting / standing merged into `still`:** orientation-robust features cannot separate postures, and posture would not transfer across devices anyway. Deployed classes: `still, walking, stairs, running, cycling`.
- Random Forests can't be converted to TFLite → small **MLPs on the same 17 features** (activity 17→32→16→5; fall 17→24→12→2), INT8-quantised.
- Bug found and fixed: early stopping on an *unweighted* validation loss made the first fall network always predict "no fall" (recall 0). Fall net now trains for a fixed 120 epochs with capped class weights.
- Post-impact **stillness check** and **cooldown** added to the live script to cut false alarms.

## 4. Current results (INT8 models, unseen subjects/users)

**Activity (single hold-out of ≈25 % of subjects):** float 0.900 → INT8 0.899 (98.8 % agreement).

| Abstain threshold | Coverage | Accuracy on accepted |
|---|---|---|
| 0.5 | 0.96 | 0.915 |
| 0.7 (**chosen**) | 0.88 | 0.943 |
| 0.9 | 0.74 | 0.971 |

**Fall — decided per recording, 5-fold over users (INT8):**

| Rule | Fall recall | ADL false alarm |
|---|---|---|
| thr 0.5, no stillness check | 0.932 | 0.282 |
| thr 0.7, no stillness check | 0.898 | 0.207 |
| thr 0.7 + stillness < 2.0 | 0.871 | 0.155 |
| thr 0.9, no stillness check | 0.865 | 0.141 |
| **thr 0.95 + stillness < 3.0 (chosen)** | **0.812** | **0.075** |

The stillness rule mostly moves along the same recall/false-alarm curve as raising the threshold. Model sizes: activity 5.1 KB, fall 4.3 KB.

## 5. Known limitations
- Public-data validation only; no test on the SiWG917 board's own IMU yet.
- BITS-2 sampling rate uncertain (fall window = 40 samples assumed 20 Hz).
- Fall detection still false-alarms on ≈8 % of ADL recordings at the chosen setting and misses ≈19 % of falls.
- Posture (lying/sitting/standing) is not distinguished.
- Each model was evaluated on one train/test split or 5-fold CV; subject counts are small (PAMAP2: 9, BITS-2: 41 users).

## 6. Repository layout
```
src/
  har_common.py            feature extraction (17 features/window) + dataset loaders (shared by training and live inference)
  export_tflite_int8.py    trains MLPs, INT8 export, float-vs-INT8 check, per-recording fall evaluation, golden vectors
  train_final_models.py    scikit-learn Random Forest version (calibrated) – baseline / --engine rf
  live_inference.py        reads ax,ay,az,gx,gy,gz from the board over serial, runs the models, prints alerts
models/                    activity_int8.tflite, fall_int8.tflite, export_params.json, golden_vectors.json, *.cc  (ONE export run – keep together)
notebooks/                 exploration notebooks (PAMAP2 / BITS-2 / MHEALTH comparison)
```

## 7. How to reproduce
**Train + export (Google Colab):** upload `src/har_common.py` and `src/export_tflite_int8.py` to `/content`, then
`!python /content/export_tflite_int8.py` (run as a fresh process; restart the runtime after editing `har_common.py`).
Copy the resulting files into `models/`.

**Live test on a laptop** (board prints `ax,ay,az,gx,gy,gz` at 100 Hz over USB-serial):
```
pip install -r requirements.txt
python src/live_inference.py --engine tflite --port COM5 --fs 100 --log sessions/s1.csv \
   --params models/export_params.json --act_model models/activity_int8.tflite --fall_model models/fall_int8.tflite
```
Options: `--acc_unit ms2`, `--gyro_unit rads`, `--thr`, `--still_std`, `--cooldown`. Raw logs go in `sessions/` (git-ignored: personal data).

## 8. Hardware notes (SiWG917)
- The SiWG917 is the wireless SoC; motion data comes from an external/board IMU. The SiWx917 Dev Kit (`SiWx917-DK2605A`) lists a TDK ICM-40627 6-axis IMU; the Pro Kit lists no motion sensor. **Board in use: _to be filled in_.**
- Board's temperature sensor measures the board/ambient, not the wearer – not used as a feature.
- Target IMU setup: 100 Hz output, accelerometer ±8 g, gyro ±1000 dps, worn on the wrist; convert to m/s² and rad/s. Check: still board → |acc| ≈ 9.81.
- On-chip pipeline: 17 window features in C → standardise (`export_params.json`) → quantise (input scale/zero-point) → TFLite Micro → dequantise → abstain / fall rules.

## 9. Roadmap
- [x] Cross-dataset analysis, fixes, honest negative results
- [x] Activity + fall models, INT8 export, golden test vectors
- [ ] Laptop test with a person wearing the board (activities, false-alarm count)
- [ ] 17-feature extraction in C, verified against `models/golden_vectors.json`
- [ ] TFLite Micro on the SiWG917, end-to-end check
- [ ] Phase 2: record own-device data, re-calibrate / fine-tune, test falls safely (crash mat)

## 10. Data sources & credits
- **PAMAP2 Physical Activity Monitoring** – A. Reiss and D. Stricker, ISWC 2012 (UCI Machine Learning Repository).
- **BITS-2** – *Inertial measurement and heart-rate sensor-based dataset for geriatric fall detection using custom built wrist-worn device*, Data in Brief (2024).
- **MHEALTH** – O. Banos et al., 2014 (UCI Machine Learning Repository).
Please cite the original authors if you use these datasets, and check each dataset's licence before redistributing anything derived from it.
