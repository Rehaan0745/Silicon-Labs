"""Run in Colab AFTER uploading har_common.py to /content :
       !python /content/train_final_models.py
   Produces /content/har_bundle.joblib (activity model + fall model + thresholds)."""
import sys, warnings, joblib, numpy as np, pandas as pd, sklearn
sys.path.insert(0, "/content"); sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
import har_common as hc
warnings.filterwarnings("ignore")
OUT = "/content/har_bundle.joblib" if __import__("os").path.isdir("/content") else "har_bundle.joblib"

RF = lambda n=200: RandomForestClassifier(n_estimators=n, min_samples_leaf=2, class_weight="balanced_subsample",
                                          random_state=42, n_jobs=-1)
def make_cal(X, y, g):
    try:
        return CalibratedClassifierCV(RF(), method="isotonic", cv=list(GroupKFold(4).split(X, y, g))).fit(X, y)
    except Exception as e:                       # a class missing in an inner fold -> fall back to raw RF probabilities
        print("  (calibration fallback:", str(e)[:60], ")"); return RF().fit(X, y)

# ---------------- ACTIVITY MODEL (PAMAP2) ----------------
pam = hc.build_pamap_activity()
print("PAMAP2 windows:", pam.shape, "\n", pam.label.value_counts().to_dict())
X, y, g = pam[hc.FEATS].to_numpy(), pam.label.to_numpy(), pam.subject.to_numpy()
oof_p = None; classes = None
for tr, te in GroupKFold(5).split(X, y, g):
    m = make_cal(X[tr], y[tr], g[tr])
    if oof_p is None: classes = list(m.classes_); oof_p = np.zeros((len(y), len(classes)))
    oof_p[te] = m.predict_proba(X[te])
oof_pred = np.array(classes)[oof_p.argmax(1)]; oof_conf = oof_p.max(1)
print(f"\nActivity accuracy on UNSEEN subjects: {accuracy_score(y, oof_pred):.3f}")
print(classification_report(y, oof_pred, zero_division=0))
print("Abstain table (predict only if confidence >= thr):")
rows, chosen = [], None
for thr in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    k = oof_conf >= thr
    acc = accuracy_score(y[k], oof_pred[k]) if k.any() else float("nan")
    rows.append({"thr": thr, "coverage": round(k.mean(), 3), "accuracy_on_accepted": round(acc, 3)})
    if chosen is None and acc >= 0.90: chosen = thr
print(pd.DataFrame(rows).to_string(index=False))
abstain_thr = chosen if chosen is not None else 0.7
print("Chosen abstain threshold:", abstain_thr)
act_model = make_cal(X, y, g)

# ---------------- FALL MODEL (BITS-2) ----------------
bits = hc.build_bits_falls()
ok = ((bits.kind == "adl") | (bits.impact == 1)).to_numpy()
print("\nBITS-2 windows:", bits.shape, "| impact windows:", int(bits.impact.sum()), "| ADL windows:", int((bits.kind == 'adl').sum()))
fall_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                    random_state=42, n_jobs=-1).fit(bits.loc[ok, hc.FEATS].to_numpy(), bits.loc[ok, "impact"].to_numpy())
FALL_THR = 0.4        # from your validation table: ~85% recall at ~10% ADL false alarms

joblib.dump({"act_model": act_model, "act_classes": list(act_model.classes_), "abstain_thr": abstain_thr,
             "fall_model": fall_model, "fall_thr": FALL_THR, "feats": hc.FEATS,
             "sklearn_version": sklearn.__version__,
             "act_rate": hc.ACT_RATE, "act_n": hc.ACT_N, "fall_rate": hc.FALL_RATE, "fall_n": hc.FALL_N}, OUT)
print("\nSaved", OUT, "| trained with scikit-learn", sklearn.__version__)
print("On your laptop install the SAME version:  pip install scikit-learn==" + sklearn.__version__)
