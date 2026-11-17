"""Etapa 3 · Explicabilidad con SHAP y LIME (como Liu et al., 2025).

Se reentrena Gradient Boosting con el protocolo de Liu (Modelo 2) y se explica:
  - Global: importancia media |SHAP| de las 15 variables principales.
  - Por paciente: gráfico waterfall de SHAP y explicación LIME de un paciente con sepsis del conjunto de prueba.
"""
import matplotlib

matplotlib.use("Agg")
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from config import FIGURES, RESULTS, SEED, TARGET, model2_features
from utils import load_data

LABELS = {
    "temperature": "Temperatura", "heartrate": "Frecuencia cardíaca", "resprate": "Frecuencia respiratoria",
    "o2sat": "SpO₂", "sbp": "Presión sistólica", "dbp": "Presión diastólica", "pain": "Dolor", "age": "Edad",
    "male": "Sexo masculino", "suspected_infection": "Sospecha de infección", "infection_symptoms": "Síntomas de infección",
    "fever": "Fiebre (motivo)", "BP_low": "Presión baja (motivo)", "mantal_altered": "Estado mental alterado",
    "lethargy": "Letargo", "tumor": "Tumor", "CKD": "Enfermedad renal crónica", "CHF": "Insuficiencia cardíaca",
    "diabetes": "Diabetes", "cirrhosis": "Cirrosis", "COPD": "EPOC", "hypertension": "Hipertensión",
    "CHD": "Enf. coronaria", "smoke": "Fumador", "chest_pain": "Dolor torácico", "abdominal_pain": "Dolor abdominal",
    "weakness": "Debilidad", "dyspnea": "Disnea", "fall": "Caída", "cough": "Tos", "GIB": "Sangrado digestivo",
    "jaundice": "Ictericia", "hemoptysis": "Hemoptisis", "tachycardia": "Taquicardia (motivo)", "chills": "Escalofríos",
    "fever_prehospital": "Fiebre prehospitalaria", "confusion": "Confusión", "N_V": "Náuseas o vómitos",
}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    feats = model2_features(df.columns)
    X_res, y_res = RandomUnderSampler(random_state=SEED).fit_resample(df[feats], df[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res, test_size=0.2, random_state=SEED)
    model = GradientBoostingClassifier(n_estimators=100, random_state=SEED).fit(X_tr, y_tr)
    names = [LABELS.get(f, f) for f in feats]

    # --- SHAP global ---
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_te.rename(columns=dict(zip(feats, names))))
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=names).sort_values(ascending=False)
    imp.round(4).to_csv(RESULTS / "03_shap_importancia_global.csv", header=["mean_abs_shap"])
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(top.index, top.values, color="#FF3A00")
    ax.set_xlabel("Importancia media |SHAP|")
    ax.set_title("SHAP global · Gradient Boosting (Modelo 2)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_global_gb.png", dpi=200)
    plt.close(fig)

    # --- Un paciente con sepsis ---
    probs = model.predict_proba(X_te)[:, 1]
    pos = np.where((y_te.values == 1) & (probs > 0.7))[0]
    i = int(pos[0]) if len(pos) else int(np.argmax(probs))
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    plt.title(f"Paciente {i} · probabilidad de sepsis {probs[i]:.0%}", fontsize=11)
    plt.gcf().set_size_inches(8, 5.2)
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_paciente_gb.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- LIME para el mismo paciente ---
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_tr.values, feature_names=names, class_names=["No sepsis", "Sepsis"],
        discretize_continuous=True, random_state=SEED)
    exp = lime_exp.explain_instance(X_te.values[i], model.predict_proba, num_features=10)
    items = exp.as_list()
    pd.DataFrame(items, columns=["condicion", "peso"]).to_csv(RESULTS / "03_lime_paciente.csv", index=False)
    lab, w = zip(*items[::-1])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(lab, w, color=["#FF3A00" if v > 0 else "#595959" for v in w])
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_title(f"LIME · paciente {i} (rojo sube, gris baja el riesgo)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "lime_paciente_gb.png", dpi=200)
    plt.close(fig)

    shap_top = pd.Series(sv.values[i], index=names).sort_values(key=abs, ascending=False).head(10)
    shap_top.round(4).to_csv(RESULTS / "03_shap_paciente.csv", header=["shap"])
    print(f"Paciente {i}: probabilidad {probs[i]:.3f}, real={y_te.values[i]}")
    print("Top SHAP global:\n", imp.head(10).round(4).to_string())
    print("SHAP paciente:\n", shap_top.round(3).to_string())
    print("LIME paciente:\n", "\n".join(f"  {a}: {b:+.3f}" for a, b in items))


if __name__ == "__main__":
    main()
