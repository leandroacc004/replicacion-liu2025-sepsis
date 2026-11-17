"""Etapa 2 · Variación propia (enfoque de mi propuesta de tesis).

Cambios frente a Liu et al. (2025):
  1. Se divide PRIMERO el dataset real, estratificado: 70% entrenamiento, 15% validación, 15% prueba.
     La prueba conserva la prevalencia real de sepsis (~5.7%), como en una emergencia.
  2. El balanceo se aplica SOLO al entrenamiento: sin balanceo, submuestreo (RUS) o SMOTE.
  3. Se agrega XGBoost a los modelos de referencia (LR, RF y GB).
  4. Se reporta AUPRC, sensibilidad y precisión de la clase sepsis, y un umbral elegido en validación
     (máximo F1), además del umbral 0.5.
"""
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import RESULTS, SEED, TARGET, model2_features
from utils import load_data, metrics


def models():
    return {
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    }


def best_threshold(y_val, p_val):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])])


def main():
    df = load_data()
    feats = model2_features(df.columns)
    X, y = df[feats], df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f"Entrenamiento={len(y_tr)}  Validación={len(y_va)}  Prueba={len(y_te)}  sepsis en prueba={y_te.mean():.2%}")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = (scaler.transform(a) for a in (X_tr, X_va, X_te))
    samplers = {
        "Sin balanceo": None,
        "Submuestreo (RUS)": RandomUnderSampler(random_state=SEED),
        "SMOTE": SMOTE(random_state=SEED),
    }
    rows = []
    for s_name, sampler in samplers.items():
        Xb, yb = (X_tr_s, y_tr) if sampler is None else sampler.fit_resample(X_tr_s, y_tr)
        print(f"\n[{s_name}] filas de entrenamiento={len(yb)}  sepsis={np.mean(yb):.1%}")
        for m_name, model in models().items():
            t0 = time.time()
            model.fit(Xb, yb)
            p_va = model.predict_proba(X_va_s)[:, 1]
            p_te = model.predict_proba(X_te_s)[:, 1]
            thr = best_threshold(y_va, p_va)
            for thr_name, t in (("0.5", 0.5), ("Óptimo en validación", thr)):
                m = metrics(y_te, p_te, threshold=t)
                m.update({"Balanceo": s_name, "Algoritmo": m_name, "Umbral": thr_name,
                          "Valor_umbral": round(t, 3), "Segundos": round(time.time() - t0, 1)})
                rows.append(m)
            r = rows[-2]
            print(f"  {m_name:20s} AUC={r['AUC']:.3f} AUPRC={r['AUPRC']:.3f} Sens={r['Sensitivity']:.3f} "
                  f"Prec={r['Precision']:.3f} F1={r['F1']:.3f}  ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.round(4).to_csv(RESULTS / "02_variacion_tesis.csv", index=False)


if __name__ == "__main__":
    main()
