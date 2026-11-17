"""Funciones compartidas: carga de datos, modelos y métricas."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import DATA_CSV, DATA_XLSX, SEED


def load_data():
    """Lee el dataset de Liu et al. (figshare). La primera vez convierte el Excel a CSV."""
    if not DATA_CSV.exists():
        if not DATA_XLSX.exists():
            raise FileNotFoundError("Falta data/data_xlsx.xlsx. Ejecuta primero src/00_descargar_datos.py")
        pd.read_excel(DATA_XLSX, engine="openpyxl").to_csv(DATA_CSV, index=False)
    return pd.read_csv(DATA_CSV)


def liu_models():
    """Los 8 algoritmos con los mismos hiperparámetros del código suplementario de Liu et al."""
    return {
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=SEED),
        "Naive Bayes": GaussianNB(),
    }


def metrics(y_true, y_prob, threshold=0.5):
    """Métricas binarias para la clase sepsis (1)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Brier": brier_score_loss(y_true, y_prob),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }
