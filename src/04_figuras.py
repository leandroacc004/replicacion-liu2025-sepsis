"""Etapa 4 · Figuras de comparación para la presentación."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, RESULTS

RED, GRAY, LIGHT = "#FF3A00", "#595959", "#BFBFBF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def comparacion_auc():
    r = pd.read_csv(RESULTS / "01_replicacion_liu.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, modelo in zip(axes, ["Modelo 1", "Modelo 2"]):
        d = r[r.Modelo == modelo].sort_values("AUC_Liu", ascending=False)
        x = np.arange(len(d))
        ax.bar(x - 0.2, d.AUC_Liu, 0.4, color=LIGHT, label="Liu et al. (2025)")
        ax.bar(x + 0.2, d.AUC, 0.4, color=RED, label="Mi réplica")
        ax.set_xticks(x, [a.replace("Logistic Regression", "LR").replace("Gradient Boosting", "GB")
                          .replace("Random Forest", "RF").replace("Extra Trees", "ET").replace("Decision Tree", "DT")
                          .replace("Naive Bayes", "NB") for a in d.Algoritmo])
        ax.set_ylim(0.5, 0.9)
        ax.set_title(f"{modelo} · AUC en prueba")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUC")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "comparacion_auc_liu_vs_replica.png", dpi=200)
    plt.close(fig)


def balanceada_vs_real():
    r1 = pd.read_csv(RESULTS / "01_replicacion_liu.csv")
    r2 = pd.read_csv(RESULTS / "02_variacion_tesis.csv")
    a = r1[(r1.Modelo == "Modelo 2") & (r1.Algoritmo == "Gradient Boosting")].iloc[0]
    b = r2[(r2.Balanceo == "Submuestreo (RUS)") & (r2.Algoritmo == "Gradient Boosting") & (r2.Umbral == "0.5")].iloc[0]
    mets = ["AUC", "Sensitivity", "Precision", "F1"]
    lab = ["AUC", "Sensibilidad", "Precisión", "F1"]
    x = np.arange(len(mets))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.2, [a[m] for m in mets], 0.4, color=LIGHT, label="Prueba balanceada 50/50 (protocolo de Liu)")
    ax.bar(x + 0.2, [b[m] for m in mets], 0.4, color=RED, label="Prueba con prevalencia real 5.7%")
    for i, m in enumerate(mets):
        ax.text(i - 0.2, a[m] + 0.02, f"{a[m]:.2f}", ha="center", fontsize=9)
        ax.text(i + 0.2, b[m] + 0.02, f"{b[m]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, lab)
    ax.set_ylim(0, 1)
    ax.set_title("Gradient Boosting: mismo modelo, distinta prueba")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "prueba_balanceada_vs_real.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparacion_auc()
    balanceada_vs_real()
    print("Figuras guardadas en", FIGURES)
