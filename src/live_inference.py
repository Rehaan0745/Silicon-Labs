"""Live test on your laptop. The board prints one line per sample over USB-serial:   ax,ay,az,gx,gy,gz
   python live_inference.py --engine tflite --port COM5 --fs 100 --log session1.csv     <- tests the EXACT INT8 models for the chip
   python live_inference.py --engine rf     --port COM5 --fs 100                          <- scikit-learn Random Forest bundle
   python live_inference.py --engine tflite --file session1.csv --fs 100                  <- replay a saved log
Needs: pip install pyserial numpy scipy joblib pandas  (+ tensorflow for --engine tflite, scikit-learn==<training version> for rf)
Files: har_common.py, and either export_params.json + activity_int8.tflite + fall_int8.tflite (tflite) or har_bundle.joblib (rf)."""
import argparse, collections, json, numpy as np
from scipy.signal import resample_poly
from har_common import window_features

G = 9.80665
ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["tflite", "rf"], default="tflite")
ap.add_argument("--bundle", default="har_bundle.joblib"); ap.add_argument("--params", default="export_params.json")
ap.add_argument("--act_model", default="activity_int8.tflite"); ap.add_argument("--fall_model", default="fall_int8.tflite")
ap.add_argument("--port"); ap.add_argument("--baud", type=int, default=115200)
ap.add_argument("--file"); ap.add_argument("--log")
ap.add_argument("--fs", type=int, default=100, help="IMU sample rate printed by the board (Hz)")
ap.add_argument("--acc_unit", choices=["g", "ms2"], default="g"); ap.add_argument("--gyro_unit", choices=["dps", "rads"], default="dps")
ap.add_argument("--thr", type=float, help="override fall probability threshold")
ap.add_argument("--still_std", type=float, help="override post-impact stillness limit (m/s^2); 0 disables")
ap.add_argument("--cooldown", type=float, default=10.0, help="seconds between fall alerts")
a = ap.parse_args()

# ------------------------------------------------------------------ engines (same interface)
class RFEngine:
    def __init__(self):
        import joblib; B = joblib.load(a.bundle); self.B = B
        self.rate, self.n, self.frate, self.fn = B["act_rate"], B["act_n"], B["fall_rate"], B["fall_n"]
        self.abstain, self.fall_thr, self.still = B["abstain_thr"], B["fall_thr"], None
    def act(self, f):
        p = self.B["act_model"].predict_proba([f])[0]; k = int(p.argmax()); return self.B["act_classes"][k], float(p[k])
    def fall(self, f): return float(self.B["fall_model"].predict_proba([f])[0, 1])

class TFLiteEngine:
    def __init__(self):
        P = json.load(open(a.params)); self.P = P
        try: from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf; Interpreter = tf.lite.Interpreter
        self.m = {}
        for name, path in (("activity", a.act_model), ("fall", a.fall_model)):
            it = Interpreter(model_path=path); it.allocate_tensors()
            self.m[name] = (it, it.get_input_details()[0], it.get_output_details()[0])
        self.rate, self.n, self.frate, self.fn = P["act_rate_hz"], P["act_window"], P["fall_rate_hz"], P["fall_window"]
        self.abstain = P["activity"].get("abstain_thr", 0.8)
        self.fall_thr = P["fall"].get("threshold", 0.7); self.still = P["fall"].get("still_std")
    def _run(self, name, f):
        it, i, o = self.m[name]; Pm = self.P[name]
        z = (np.asarray(f) - np.array(Pm["mean"])) / np.array(Pm["std"])
        q = np.clip(np.round(z / Pm["in_scale"] + Pm["in_zero_point"]), -128, 127).astype(np.int8)[None, :]
        it.set_tensor(i["index"], q); it.invoke()
        return (it.get_tensor(o["index"])[0].astype(np.float32) - Pm["out_zero_point"]) * Pm["out_scale"]
    def act(self, f):
        p = self._run("activity", f); k = int(p.argmax()); return self.P["activity"]["classes"][k], float(p[k])
    def fall(self, f): return float(self._run("fall", f)[1])

E = RFEngine() if a.engine == "rf" else TFLiteEngine()
THR = a.thr if a.thr is not None else E.fall_thr
STILL = (None if a.still_std == 0 else a.still_std) if a.still_std is not None else E.still
acc_k = G if a.acc_unit == "g" else 1.0; gyro_k = np.pi / 180 if a.gyro_unit == "dps" else 1.0
print(f"engine={a.engine} | fall thr={THR} | post-impact stillness={'off' if STILL is None else f'|acc| std < {STILL} m/s^2'} "
      f"| cooldown={a.cooldown}s | abstain thr={E.abstain}", flush=True)

# ------------------------------------------------------------------ streaming logic
buf = collections.deque(maxlen=2 * a.fs); hop = a.fs // 2
n_in, mags, last_alarm, pending = 0, [], -1e9, None

def features(w):
    w = np.asarray(w, float).copy(); w[:, :3] *= acc_k; w[:, 3:] *= gyro_k
    act = resample_poly(w, E.rate, a.fs, axis=0)[: E.n]; fal = resample_poly(w, E.frate, a.fs, axis=0)[: E.fn]
    return (window_features(act[:, :3], act[:, 3:]), window_features(fal[:, :3], fal[:, 3:]),
            float(np.linalg.norm(w[:, :3], axis=1).mean()))

def recent_motion():                       # std of |acc| over the last second, m/s^2
    w = np.array(list(buf)[-a.fs:])[:, :3] * acc_k
    return float(np.linalg.norm(w, axis=1).std())

def handle(vals):
    global n_in, last_alarm, pending
    buf.append(vals); n_in += 1; t = n_in / a.fs
    if pending is not None and n_in >= pending:                      # post-impact check, 2 s after the flagged window
        mot = recent_motion(); pending = None
        if mot < STILL: last_alarm = t; print(f"t={t:7.1f}s  *** FALL ALERT ***  (impact then stillness: motion std {mot:.2f})", flush=True)
        else:           print(f"t={t:7.1f}s  fall flag dismissed: still moving (motion std {mot:.2f})", flush=True)
    if len(buf) == buf.maxlen and n_in % hop == 0:
        fa, ff, mag = features(list(buf)); mags.append(mag)
        label, conf = E.act(fa); label = label if conf >= E.abstain else "uncertain"; pf = E.fall(ff)
        flagged = pf >= THR and pending is None and (t - last_alarm) >= a.cooldown
        tag = ""
        if flagged:
            if STILL is None: last_alarm = t; tag = "  *** FALL ALERT ***"
            else: pending = n_in + 2 * a.fs; tag = "  <fall flagged, checking stillness...>"
        print(f"t={t:7.1f}s  activity={label:10s} conf={conf:.2f}  p_fall={pf:.2f}  |acc|={mag:5.2f}{tag}", flush=True)
        if len(mags) == 10 and not (6.0 < np.median(mags) < 14.0):
            print("!! WARNING: mean |acc| is far from 9.81 m/s^2 -> check --acc_unit / sensor range / axis order", flush=True)

def lines():
    if a.file: yield from open(a.file)
    else:
        import serial; s = serial.Serial(a.port, a.baud, timeout=1)
        while True: yield s.readline().decode(errors="ignore")

log = open(a.log, "w") if a.log else None
if log: log.write("ax,ay,az,gx,gy,gz\n")
try:
    for line in lines():
        parts = line.strip().split(",")
        if len(parts) != 6: continue
        try: vals = [float(x) for x in parts]
        except ValueError: continue
        if log: log.write(line.strip() + "\n")
        handle(vals)
except KeyboardInterrupt:
    pass
finally:
    if log: log.close()
