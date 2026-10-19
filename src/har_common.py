"""Shared code: feature extraction + dataset loaders. Used by training AND live inference so they never disagree."""
import os, re
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import resample_poly

PAMAP_DIR = Path(os.environ.get("PAMAP_DIR", "/content/PAMAP2_Data/PAMAP2_Dataset/Protocol"))
BITS_DIR  = Path(os.environ.get("BITS_DIR",  "/content/BITS-2/Dataset"))

ACT_RATE,  ACT_N,  ACT_STRIDE  = 50, 100, 50      # activity: 2 s windows at 50 Hz
FALL_RATE, FALL_N, FALL_STRIDE = 20, 40, 20       # fall: BITS-2 nominal 20 Hz, 40-sample windows

# PAMAP2 activity id -> deployment class. Lying/sitting/standing are merged into "still": posture cannot be
# told apart reliably with orientation-free features, and it would not transfer across devices anyway.
ACT_CLASSES = {1:"still", 2:"still", 3:"still", 4:"walking", 5:"running", 6:"cycling", 12:"stairs", 13:"stairs"}

FEATS = (["acc_mag_mean","acc_mag_std","acc_mag_min","acc_mag_max","acc_mag_p10","acc_mag_p90"]
       + ["gyr_mag_mean","gyr_mag_std","gyr_mag_max","gyr_mag_p10","gyr_mag_p90"]
       + ["acc_std_1","acc_std_2","acc_std_3","gyr_std_1","gyr_std_2","gyr_std_3"])

def window_features(acc, gyr):
    """acc (n,3) in m/s^2 INCLUDING gravity, gyr (n,3) in rad/s. Orientation-robust statistics."""
    am = np.linalg.norm(acc, axis=1); gm = np.linalg.norm(gyr, axis=1)
    f = [am.mean(), am.std(), am.min(), am.max(), *np.percentile(am, [10, 90]),
         gm.mean(), gm.std(), gm.max(), *np.percentile(gm, [10, 90])]
    f += sorted(acc.std(axis=0), reverse=True)
    f += sorted(gyr.std(axis=0), reverse=True)
    return f

def _runs(labels):
    cuts = np.r_[0, np.flatnonzero(np.diff(labels) != 0) + 1, len(labels)]
    return zip(cuts[:-1], cuts[1:])

def build_pamap_activity():
    rows = []
    for f in sorted(PAMAP_DIR.glob("subject*.dat")):
        subj = f.stem.replace("subject", "")
        df = pd.read_csv(f, sep=r"\s+", header=None, usecols=[1,4,5,6,10,11,12]).apply(pd.to_numeric, errors="coerce")
        act = df[1].fillna(0).astype(int).to_numpy()
        imu = df[[4,5,6,10,11,12]].interpolate(limit_direction="both").to_numpy(float)
        for s, e in _runs(act):
            if act[s] in ACT_CLASSES and e - s >= 100:
                x = resample_poly(imu[s:e], ACT_RATE, 100, axis=0)
                for i in range(0, len(x) - ACT_N + 1, ACT_STRIDE):
                    w = x[i:i+ACT_N]
                    rows.append(window_features(w[:, :3], w[:, 3:]) + [ACT_CLASSES[act[s]], subj])
    return pd.DataFrame(rows, columns=FEATS + ["label", "subject"])

def read_bits_streams(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    s = df["Unnamed: 5"].str.strip().str.lower(); out = {}
    for name in ("acg", "gyro"):
        v = df.loc[s == name, ["x","y","z"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
        out[name] = v[np.isfinite(v).all(axis=1)]
    return out

def build_bits_falls(n=FALL_N, stride=FALL_STRIDE):
    """Impact-centred labels: only windows containing the acceleration peak of a fall recording are positives."""
    rows = []
    for kind in ("adl", "fall"):
        for p in sorted((BITS_DIR / kind).rglob("*.csv")):
            st = read_bits_streams(p); m = min(len(st["acg"]), len(st["gyro"]))
            if m < n: continue
            a, g = st["acg"][:m], st["gyro"][:m]
            peak = int(np.argmax(np.linalg.norm(a, axis=1))) if kind == "fall" else -1
            for k, i in enumerate(range(0, m - n + 1, stride)):
                impact = int(kind == "fall" and i <= peak < i + n)
                rows.append(window_features(a[i:i+n], g[i:i+n]) + [kind, p.parent.name, p.name, k, impact])
    return pd.DataFrame(rows, columns=FEATS + ["kind", "user", "file", "win", "impact"])


def golden_raw_windows():
    """A few REAL raw windows (columns: ax,ay,az m/s^2, gx,gy,gz rad/s) so firmware can be checked against Python."""
    out = {"activity": [], "fall": []}; seen = set(); want = set(ACT_CLASSES.values())
    for f in sorted(PAMAP_DIR.glob("subject*.dat")):
        df = pd.read_csv(f, sep=r"\s+", header=None, usecols=[1,4,5,6,10,11,12]).apply(pd.to_numeric, errors="coerce")
        act = df[1].fillna(0).astype(int).to_numpy()
        imu = df[[4,5,6,10,11,12]].interpolate(limit_direction="both").to_numpy(float)
        for s, e in _runs(act):
            lbl = ACT_CLASSES.get(act[s])
            if lbl and lbl not in seen and e - s >= 400:
                x = resample_poly(imu[s:e], ACT_RATE, 100, axis=0); mid = len(x) // 2
                if len(x[mid:mid + ACT_N]) == ACT_N:
                    out["activity"].append({"label": lbl, "window": x[mid:mid + ACT_N].tolist()}); seen.add(lbl)
        if seen == want: break
    for kind in ("fall", "adl"):
        for p in sorted((BITS_DIR / kind).rglob("*.csv")):
            st = read_bits_streams(p); m = min(len(st["acg"]), len(st["gyro"]))
            if m < FALL_N: continue
            a, g = st["acg"][:m], st["gyro"][:m]
            i = 0 if kind == "adl" else max(0, min(int(np.argmax(np.linalg.norm(a, axis=1))) - FALL_N // 2, m - FALL_N))
            out["fall"].append({"kind": kind, "window": np.hstack([a[i:i+FALL_N], g[i:i+FALL_N]]).tolist()}); break
    return out
