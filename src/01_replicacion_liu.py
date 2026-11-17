"""Etapa 1 · Réplica fiel de Liu et al. (2025).

Protocolo del código suplementario de los autores:
  1. Submuestreo aleatorio (RandomUnderSampler) sobre TODO el dataset.
  2. División aleatoria 80% entrenamiento / 20% prueba (random_state=42).
  3. Estandarización (StandardScaler) ajustada en entrenamiento.
  4. 8 algoritmos con hiperparámetros por defecto; umbral 0.5.
Se corre para el Modelo 1 (signos vitales) y el Modelo 2 (todas las variables del triage).
"""
import time

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import PAPER_MODEL1_AUC, PAPER_MODEL2, RESULTS, SEED, TARGET, VITALS, model2_features
from utils import liu_models, load_data, metrics


def run(df, features, label):
    X, y = df[features], df[TARGET]
    X_res, y_res = RandomUnderSampler(random_state=SEED).fit_resample(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res, test_size=0.2, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    print(f"[{label}] variables={len(features)}  entrenamiento={len(y_tr)}  prueba={len(y_te)} (balanceada)")
    rows = []
    for name, model in liu_models().items():
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        prob = model.predict_proba(X_te_s)[:, 1]
        m = metrics(y_te, prob)
        m.update({"Modelo": label, "Algoritmo": name, "Segundos": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"  {name:20s} AUC={m['AUC']:.3f} F1={m['F1']:.3f}")
    return rows


def main():
    df = load_data()
    print(f"Dataset: {df.shape[0]} pacientes, sepsis={df[TARGET].mean():.2%}")
    feats2 = model2_features(df.columns)
    rows = run(df, VITALS, "Modelo 1") + run(df, feats2, "Modelo 2")
    res = pd.DataFrame(rows)

    # Comparación con lo publicado
    res["AUC_Liu"] = [
        (PAPER_MODEL1_AUC if r.Modelo == "Modelo 1" else {k: v["AUC"] for k, v in PAPER_MODEL2.items()})[r.Algoritmo]
        for r in res.itertuples()
    ]
    for k in ["F1", "Accuracy", "Sensitivity", "Specificity"]:
        res[f"{k}_Liu"] = [PAPER_MODEL2[r.Algoritmo][k] if r.Modelo == "Modelo 2" else None for r in res.itertuples()]
    res["Dif_AUC"] = res["AUC"] - res["AUC_Liu"]
    RESULTS.mkdir(exist_ok=True)
    res.round(4).to_csv(RESULTS / "01_replicacion_liu.csv", index=False)
    pd.Series(feats2).to_csv(RESULTS / "variables_modelo2.csv", index=False, header=["variable"])
    print(res[["Modelo", "Algoritmo", "AUC", "AUC_Liu", "Dif_AUC", "F1", "F1_Liu"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
