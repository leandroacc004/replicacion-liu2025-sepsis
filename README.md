# Réplica de Liu et al. (2025): ML interpretable para el riesgo de sepsis en el triage

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leandroacc004/replicacion-liu2025-sepsis-triage/blob/main/Replicacion_Liu2025.ipynb)

Proyecto del curso **Propuesta de Investigación** (Ingeniería de Sistemas, Universidad de Lima, 2026-2).
Autor: Leandro Aaron Caramantin Cruz.

Este repositorio replica el artículo:

> Liu, Z., Shu, W., Li, T., Zhang, X., & Chong, W. (2025). Interpretable machine learning for predicting sepsis risk in emergency triage patients. *Scientific Reports, 15*, 887. https://doi.org/10.1038/s41598-025-85121-z

Además agrega una **variación propia**, alineada con mi propuesta de tesis *"Modelo de aprendizaje automático explicable para apoyar el triage en servicios de emergencia"*.

## Workflow

| Etapa | Réplica fiel (Liu et al.) | Variación propia |
|---|---|---|
| 1. Datos | Dataset de los autores en figshare (MIMIC-IV-ED): 189,167 atenciones, sepsis 5.7% | — |
| 2. Preprocesamiento | Modelo 1: 7 signos vitales. Modelo 2: 78 variables (vitales, edad, sexo, antecedentes y motivo de consulta) | Exclusión explícita de variables con fuga de información |
| 3. Desbalance y partición | Submuestreo aleatorio de **todo** el dataset y división 80/20 (prueba 50/50) | División 70/15/15 estratificada; submuestreo o SMOTE **solo en entrenamiento**; prueba con la prevalencia real |
| 4. Modelado | LR, árbol de decisión, Extra Trees, Gradient Boosting, Random Forest, KNN, SVM y Naive Bayes, con los parámetros del código suplementario | XGBoost; umbral elegido en validación (máximo F1) |
| 5. Evaluación | AUC, F1, accuracy, sensibilidad y especificidad | AUPRC y precisión con la prevalencia real |
| 6. Explicabilidad | SHAP global y por paciente; LIME por paciente (Gradient Boosting) | — |

## Cómo ejecutarlo

**Opción 1 · Google Colab (recomendada):** abre `Replicacion_Liu2025.ipynb` con el botón **Abrir en Colab** y usa *Entorno de ejecución → Ejecutar todas*. El cuaderno instala las librerías, descarga el dataset y corre todo el workflow con explicaciones. Ya viene ejecutado, así que los resultados se ven sin correrlo.

**Opción 2 · Scripts en tu computadora:**

```bash
pip install -r requirements.txt
cd src
python 00_descargar_datos.py      # descarga el dataset de figshare y lo convierte a CSV
python 01_replicacion_liu.py      # réplica fiel (Modelo 1 y Modelo 2, 8 algoritmos)
python 02_variacion_tesis.py      # variación: prueba con prevalencia real, SMOTE/RUS, XGBoost
python 03_explicabilidad.py       # SHAP y LIME
python 04_figuras.py              # figuras de comparación
```

En una laptop normal tarda en total unos 10 minutos. Los resultados se guardan en `results/` y las figuras en `results/figures/`.

`pip install lime` puede fallar con Python 3.12 o superior. Si pasa, usa Python 3.10 o 3.11, o Google Colab.

## Resultados

### Réplica fiel: AUC en el conjunto de prueba

| Algoritmo | Modelo 1 Liu | Modelo 1 mío | Modelo 2 Liu | Modelo 2 mío |
|---|---|---|---|---|
| Gradient Boosting | 0.76 | 0.756 | 0.83 | 0.806 |
| Random Forest | 0.74 | 0.738 | 0.82 | 0.803 |
| SVM | 0.75 | 0.754 | 0.82 | 0.801 |
| Extra Trees | 0.73 | 0.736 | 0.82 | 0.798 |
| Regresión logística | 0.72 | 0.720 | 0.79 | 0.794 |
| Naive Bayes | 0.74 | 0.741 | 0.74 | 0.731 |
| KNN | 0.68 | 0.685 | 0.76 | 0.737 |
| Árbol de decisión | 0.60 | 0.607 | 0.65 | 0.639 |

- **Modelo 1:** los 8 algoritmos quedan a menos de 0.01 de lo publicado.
- **Modelo 2:** mis resultados quedan entre 0.01 y 0.02 por debajo, salvo la regresión logística. El artículo no publica la lista exacta de columnas que usó, y en mi caso excluí todas las que se conocen después del triage (ver Notas).

### Hallazgo: la prueba balanceada infla la precisión

Liu submuestrea el dataset **antes** de dividir, así que su prueba tiene 50% de sepsis. Con el mismo Gradient Boosting, entrenado con submuestreo pero probado con la prevalencia real (5.7%), estos son los resultados:

| Métrica | Prueba 50/50 (Liu) | Prueba real 5.7% |
|---|---|---|
| AUC | 0.81 | 0.81 |
| Sensibilidad | 0.70 | 0.72 |
| Precisión | 0.74 | **0.15** |
| F1 | 0.72 | **0.25** |

El AUC no cambia, pero en un servicio real solo 15 de cada 100 alertas de sepsis serían correctas.

### Variación: prueba real con el umbral elegido en validación

| Balanceo | Algoritmo | AUC | AUPRC | Sensibilidad | Precisión | F1 |
|---|---|---|---|---|---|---|
| Sin balanceo | **XGBoost** | **0.818** | **0.276** | 0.397 | 0.279 | **0.328** |
| Sin balanceo | Gradient Boosting | 0.811 | 0.271 | 0.325 | 0.305 | 0.315 |
| Submuestreo | XGBoost | 0.815 | 0.265 | 0.411 | 0.268 | 0.324 |
| Submuestreo | Gradient Boosting | 0.810 | 0.264 | 0.396 | 0.271 | 0.322 |
| SMOTE | XGBoost | 0.801 | 0.246 | 0.428 | 0.239 | 0.307 |
| SMOTE | Gradient Boosting | 0.778 | 0.202 | 0.375 | 0.213 | 0.272 |

La tabla completa está en `results/02_variacion_tesis.csv`, con regresión logística, Random Forest y el umbral de 0.5.

- XGBoost logra el mejor AUC y AUPRC.
- Balancear no mejora el AUC; solo mueve el punto de corte.
- SMOTE no ayudó a los modelos de árboles. Elegir el umbral en validación es más efectivo.

### Explicabilidad

- **SHAP global** (Gradient Boosting, Modelo 2): las variables más influyentes son presión sistólica, edad, frecuencia cardíaca, frecuencia respiratoria, SpO₂, enfermedad renal crónica y cirrosis.
- **Paciente de ejemplo** (probabilidad de sepsis del 74%): SHAP y LIME coinciden en que la **cirrosis** y la **presión sistólica baja** son lo que más sube el riesgo.

Figuras: `shap_global_gb.png`, `shap_paciente_gb.png`, `lime_paciente_gb.png`.

## Notas

- **Prevalencia:** el dataset de figshare tiene 189,167 filas (10,824 con sepsis, 5.72%). El artículo dice 189,617 y 5.95%, pero sus propios conteos por grupo (178,343 + 10,824) suman 189,167.
- **Fuga de información:** el código suplementario usa como predictores todas las columnas excepto `hadm_id`, `sepsis`, `icu_los`, `outcome`, `death` e `icu`. El archivo de figshare trae además columnas que se conocen después del triage (`sepsis_hos`, `septic_shock`, `morta_28`, `hospital_expire_flag`, fechas). Si se incluyen, el AUC sube a 0.99. Por eso se excluyen (ver `src/config.py`).
- **Columnas derivadas:** también se excluyen las que duplican otras variables (`fahrenheit`, `MAP`, `MAP_range`, `temperature_range`, `temperature6`, `non_suspected_infection`). Probé incluirlas y el AUC no cambia.
- **Semilla:** todas las particiones y modelos usan `random_state=42`, como en el código de los autores.

## Estructura

```
├── Replicacion_Liu2025.ipynb   # cuaderno completo para Colab (ya ejecutado)
├── data/          # el dataset se descarga aquí (no se sube al repositorio)
├── src/           # código: 00 → 04
├── results/       # métricas en CSV y figuras
└── slides/        # presentación del pitch
```

## Licencias y créditos

- **Datos:** Liu et al. (2025), figshare, doi:10.6084/m9.figshare.26098048.v1, licencia CC BY 4.0. Derivados de MIMIC-IV-ED (PhysioNet).
- **Código base:** adaptado de los materiales suplementarios del artículo.
