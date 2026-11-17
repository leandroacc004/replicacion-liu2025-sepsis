"""Configuración común: rutas, variables de cada modelo y resultados publicados por Liu et al. (2025)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_XLSX = ROOT / "data" / "data_xlsx.xlsx"
DATA_CSV = ROOT / "data" / "liu_data.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEED = 42
TARGET = "sepsis"

# Modelo 1: solo signos vitales del triage (igual que el código suplementario de Liu)
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]

# Columnas que NO pueden ser predictores:
# identificadores, desenlaces posteriores al triage (fuga de información) y fechas.
IDS = ["subject_id", "hadm_id", "stay_id"]
OUTCOMES_AND_LEAKAGE = [
    "sepsis", "sepsis_hos", "septic_shock", "icu", "icu_los", "morta_28",
    "hospital_expire_flag", "sur_time", "admittime2", "dischtime2", "deathtime2", "ed_intime2",
]
# Variables derivadas de otras ya incluidas (duplicados); se excluyen para respetar
# la lista de predictores descrita en el artículo.
DERIVED = ["fahrenheit", "MAP", "MAP_range", "temperature_range", "temperature6", "non_suspected_infection"]

EXCLUDE_M2 = IDS + OUTCOMES_AND_LEAKAGE + DERIVED


def model2_features(columns):
    """Modelo 2: signos vitales + edad y sexo + antecedentes + motivo de consulta (síntomas)."""
    return [c for c in columns if c not in EXCLUDE_M2]


# Resultados publicados en el conjunto de prueba (Liu et al., 2025, Tabla 3 y texto).
PAPER_MODEL2 = {
    "Gradient Boosting": {"AUC": 0.83, "F1": 0.75, "Accuracy": 0.75, "Sensitivity": 0.74, "Specificity": 0.78},
    "Extra Trees":       {"AUC": 0.82, "F1": 0.74, "Accuracy": 0.75, "Sensitivity": 0.74, "Specificity": 0.76},
    "Random Forest":     {"AUC": 0.82, "F1": 0.74, "Accuracy": 0.74, "Sensitivity": 0.75, "Specificity": 0.74},
    "SVM":               {"AUC": 0.82, "F1": 0.74, "Accuracy": 0.75, "Sensitivity": 0.73, "Specificity": 0.76},
    "Logistic Regression": {"AUC": 0.79, "F1": 0.72, "Accuracy": 0.73, "Sensitivity": 0.71, "Specificity": 0.73},
    "KNN":               {"AUC": 0.76, "F1": 0.68, "Accuracy": 0.70, "Sensitivity": 0.65, "Specificity": 0.76},
    "Naive Bayes":       {"AUC": 0.74, "F1": 0.71, "Accuracy": 0.71, "Sensitivity": 0.72, "Specificity": 0.70},
    "Decision Tree":     {"AUC": 0.65, "F1": 0.64, "Accuracy": 0.65, "Sensitivity": 0.65, "Specificity": 0.66},
}
PAPER_MODEL1_AUC = {
    "Gradient Boosting": 0.76, "SVM": 0.75, "Random Forest": 0.74, "Naive Bayes": 0.74,
    "Extra Trees": 0.73, "Logistic Regression": 0.72, "KNN": 0.68, "Decision Tree": 0.60,
}
