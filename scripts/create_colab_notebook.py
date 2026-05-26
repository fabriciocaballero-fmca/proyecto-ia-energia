from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "00_colab_ejecucion_completa.ipynb"
GUIDE = ROOT / "COLAB_GUIA.md"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    md(
        """# Proyecto IA - Ejecucion completa en Google Colab

Este notebook corre el avance del proyecto sin depender de rutas locales de Windows.

Flujo:

1. Subir `NuevoTPIA.csv`.
2. Limpiar la serie temporal.
3. Crear features nuevas.
4. Entrenar un modelo Ridge.
5. Hacer Grid Search.
6. Generar metricas, predicciones y graficos.
7. Descargar los resultados.

El objetivo es mostrar un avance reproducible del proyecto al profesor."""
    ),
    md(
        """## 1. Subir el CSV

Cuando ejecutes la siguiente celda, Colab te va a pedir que selecciones el archivo.

Subi este archivo:

`NuevoTPIA.csv`"""
    ),
    code(
        """from google.colab import files
uploaded = files.upload()

csv_name = next(iter(uploaded.keys()))
print("Archivo cargado:", csv_name)"""
    ),
    md("## 2. Importar librerias"),
    code(
        """import os
import json
import zipfile
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 12

os.makedirs("data_processed", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)"""
    ),
    md("## 3. Cargar y limpiar datos"),
    code(
        """TARGET = "SIN Imputed"

df = pd.read_csv(csv_name)

# La primera columna viene como fecha/hora, pero no siempre tiene nombre claro.
df = df.rename(columns={df.columns[0]: "DATETIME"})

df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True, errors="coerce")
df = df.dropna(subset=["DATETIME", TARGET])
df = df.sort_values("DATETIME")
df = df.drop_duplicates("DATETIME")
df = df.set_index("DATETIME")

# Convertimos columnas numericas.
numeric_cols = [c for c in df.columns if c != "DATETIME"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Forzamos frecuencia horaria e interpolamos huecos.
df = df.asfreq("1h")
df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")

print("Filas:", len(df))
print("Columnas:", df.columns.tolist())
df.head()"""
    ),
    md("## 4. Crear features nuevas"),
    md(
        """### Tabla de features creadas

| Grupo | Ejemplos | Idea principal |
|---|---|---|
| Calendario ciclico | `hour_sin`, `hour_cos`, `dow_sin`, `month_cos` | Representan hora, dia y mes como ciclos. |
| Comportamiento energetico | `is_weekend`, `is_night`, `is_peak_hour` | Capturan diferencias de consumo por habitos humanos. |
| Feriados | `is_holiday`, `is_day_before_holiday`, `is_day_after_holiday` | Marcan dias especiales donde el consumo puede cambiar. |
| Sensibilidad termica | `cooling_degree_24c`, `heating_degree_18c` | Miden calor/frio respecto a una zona confortable aproximada. |
| Memoria temporal | `lag_1h`, `lag_24h`, `lag_168h`, `rolling_6h_mean` | Usan informacion reciente y patrones diarios/semanales. |

Estas variables se agregan porque el consumo de energia no depende solo del clima: tambien depende de horarios, costumbres, dias laborales y memoria reciente de la serie."""
    ),
    code(
        """# Usamos horario de Paraguay para que las franjas horarias tengan sentido local.
idx = df.index.tz_convert("America/Asuncion")
hour = idx.hour.to_numpy()
dow = idx.dayofweek.to_numpy()
month = idx.month.to_numpy()
local_dates = np.array([ts.date() for ts in idx])

def easter_sunday(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month_num = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month_num, day)

def paraguay_holidays(years):
    fixed = [(1,1), (3,1), (5,1), (5,14), (5,15), (6,12), (8,15), (9,29), (12,8), (12,25)]
    out = set()
    for year in years:
        out.update(date(year, month_num, day) for month_num, day in fixed)
        easter = easter_sunday(year)
        out.add(easter - timedelta(days=3))
        out.add(easter - timedelta(days=2))
    return out

holidays = paraguay_holidays(range(idx.year.min() - 1, idx.year.max() + 2))

# Features ciclicas: representan hora, dia y mes sin cortes bruscos.
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)

# Features de comportamiento energetico.
df["is_weekend"] = (dow >= 5).astype(int)
df["is_night"] = ((hour >= 19) | (hour <= 5)).astype(int)
df["is_morning"] = ((hour >= 6) & (hour <= 11)).astype(int)
df["is_afternoon"] = ((hour >= 12) & (hour <= 18)).astype(int)
df["is_peak_hour"] = (((hour >= 18) & (hour <= 22)) | ((hour >= 6) & (hour <= 8))).astype(int)
df["is_holiday"] = np.array([d in holidays for d in local_dates], dtype=int)
df["is_day_before_holiday"] = np.array([d + timedelta(days=1) in holidays for d in local_dates], dtype=int)
df["is_day_after_holiday"] = np.array([d - timedelta(days=1) in holidays for d in local_dates], dtype=int)

# Features termicas: miden cuanto se aleja la temperatura de una zona confortable aproximada.
df["cooling_degree_24c"] = np.maximum(df["T02M"] - 24, 0)
df["heating_degree_18c"] = np.maximum(18 - df["T02M"], 0)
df["hot_afternoon"] = df["cooling_degree_24c"] * df["is_afternoon"]
df["hot_peak_hour"] = df["cooling_degree_24c"] * df["is_peak_hour"]

# Features de memoria temporal.
df["lag_1h"] = df[TARGET].shift(1)
df["lag_2h"] = df[TARGET].shift(2)
df["lag_3h"] = df[TARGET].shift(3)
df["lag_24h"] = df[TARGET].shift(24)
df["lag_48h"] = df[TARGET].shift(48)
df["lag_168h"] = df[TARGET].shift(168)
df["lag_336h"] = df[TARGET].shift(336)
df["target_diff_1h"] = df[TARGET].diff(1).shift(1)
df["temp_diff_1h"] = df["T02M"].diff(1)
df["humidity_diff_1h"] = df["RH2M"].diff(1)
df["rolling_3h_mean"] = df[TARGET].shift(1).rolling(3).mean()
df["rolling_6h_mean"] = df[TARGET].shift(1).rolling(6).mean()
df["rolling_12h_mean"] = df[TARGET].shift(1).rolling(12).mean()
df["rolling_24h_mean"] = df[TARGET].shift(1).rolling(24).mean()
df["rolling_48h_mean"] = df[TARGET].shift(1).rolling(48).mean()
df["rolling_168h_mean"] = df[TARGET].shift(1).rolling(168).mean()
df["temp_max_24h"] = df["T02M"].shift(1).rolling(24).max()
df["temp_min_24h"] = df["T02M"].shift(1).rolling(24).min()
df["target_max_24h"] = df[TARGET].shift(1).rolling(24).max()
df["target_min_24h"] = df[TARGET].shift(1).rolling(24).min()

# Features meteorologicas derivadas.
df["wind_speed_10m"] = np.sqrt(df["U10M"] ** 2 + df["V10M"] ** 2)
df["temp_humidity_interaction"] = df["T02M"] * df["RH2M"]

processed = df.dropna().copy()
processed.to_csv("data_processed/processed_dataset.csv", index_label="DATETIME")

print("Dataset procesado:", processed.shape)
processed.head()"""
    ),
    md("## 5. Separar entrenamiento y validacion"),
    code(
        """train = processed[processed.index < pd.Timestamp("2021-01-01", tz="UTC")]
test = processed[processed.index >= pd.Timestamp("2021-01-01", tz="UTC")]

print("Entrenamiento:", train.index.min(), "a", train.index.max(), "| filas:", len(train))
print("Validacion:", test.index.min(), "a", test.index.max(), "| filas:", len(test))"""
    ),
    md("## 6. Definir features, modelo y metricas"),
    md(
        """### Tipo de problema

Este es un problema de **regresion de serie temporal**, porque queremos predecir un valor numerico continuo: `SIN Imputed`.

Por eso usamos:

- RMSE
- MAE
- MAPE
- R2

No usamos precision, F1-Score ni matriz de confusion porque esas metricas son para clasificacion, por ejemplo cuando el objetivo es decidir entre clases como `alto`, `medio` o `bajo`."""
    ),
    code(
        """base_weather = ["T02M", "RH2M", "PRSS", "TPP6", "U10M", "V10M"]

behavior_features = [
    "is_weekend",
    "is_night",
    "is_morning",
    "is_afternoon",
    "is_peak_hour",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
]

thermal_features = [
    "cooling_degree_24c",
    "heating_degree_18c",
    "hot_afternoon",
    "hot_peak_hour",
]

short_memory_features = [
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_48h",
    "lag_336h",
    "target_diff_1h",
    "temp_diff_1h",
    "humidity_diff_1h",
    "rolling_3h_mean",
    "rolling_6h_mean",
    "rolling_12h_mean",
    "rolling_48h_mean",
    "temp_max_24h",
    "temp_min_24h",
    "target_max_24h",
    "target_min_24h",
]

calendar_lags = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_24h",
    "lag_168h",
]

full_engineered = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    *behavior_features,
    "lag_24h",
    "lag_168h",
    "rolling_24h_mean",
    "rolling_168h_mean",
    "wind_speed_10m",
    "temp_humidity_interaction",
    *thermal_features,
    *short_memory_features,
]

feature_sets = {
    "weather_only": base_weather,
    "weather_calendar_lags": base_weather + calendar_lags,
    "weather_behavior_lags": base_weather + behavior_features + calendar_lags,
    "weather_behavior_thermal_lags": base_weather + behavior_features + thermal_features + calendar_lags,
    "weather_short_memory": base_weather + calendar_lags + short_memory_features,
    "full_engineered": base_weather + full_engineered,
}

alphas = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]

def calculate_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2"""
    ),
    md("## 7. Ejecutar Grid Search"),
    md(
        """### Que se optimiza

El Grid Search prueba varias configuraciones:

- distintos valores de `alpha` para la regresion Ridge;
- distintos grupos de variables, desde solo clima hasta todas las features generadas.

La mejor configuracion se elige usando RMSE: cuanto menor es el RMSE, mejor es la prediccion."""
    ),
    code(
        """rows = []
best = None

y_train = train[TARGET]
y_test = test[TARGET]

for set_name, cols in feature_sets.items():
    X_train = train[cols]
    X_test = test[cols]

    for alpha in alphas:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse, mae, mape, r2 = calculate_metrics(y_test.values, pred)

        row = {
            "feature_set": set_name,
            "alpha": alpha,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE_%": mape,
            "R2": r2,
        }
        rows.append(row)

        if best is None or rmse < best["RMSE"]:
            best = {
                **row,
                "model": model,
                "features": cols,
                "predictions": pred,
            }

results = pd.DataFrame(rows).sort_values("RMSE")
results.to_csv("results/hyperparameter_results.csv", index=False)

print("Mejores resultados:")
display(results.head(10))

print("Mejor modelo:")
for k, v in best.items():
    if k not in ["model", "features", "predictions"]:
        print(k, ":", v)"""
    ),
    md("## 8. Guardar predicciones y resumen"),
    md(
        """### Evaluacion preliminar

La evaluacion se hace con datos que el modelo no vio durante el entrenamiento.

Se usa:

- entrenamiento: datos anteriores a 2021;
- validacion: datos desde 2021.

Esto respeta el orden temporal y evita entrenar con informacion del futuro."""
    ),
    code(
        """pred_df = pd.DataFrame({
    "DATETIME": test.index,
    "actual": y_test.values,
    "prediction": best["predictions"],
})
pred_df["error"] = pred_df["prediction"] - pred_df["actual"]
pred_df["absolute_error"] = pred_df["error"].abs()
pred_df.to_csv("results/predictions.csv", index=False)

summary = {k: v for k, v in best.items() if k not in ["model", "features", "predictions"]}
summary["features"] = best["features"]

with open("results/metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pred_df.head()"""
    ),
    md("## 9. Graficos principales"),
    md(
        """### Visualizaciones incluidas

Los graficos cumplen funciones distintas:

- `predictions_vs_actual.png`: muestra si la curva predicha sigue a la curva real.
- `error_histogram.png`: muestra como se distribuyen los errores.
- `rmse_by_feature_set.png`: compara que grupos de variables funcionaron mejor.

Como el modelo Ridge no entrena por epochs como una red neuronal, no tiene una curva de entrenamiento clasica. En su lugar, se evalua la convergencia/comparacion mediante Grid Search y RMSE por configuracion."""
    ),
    code(
        """# Prediccion vs valor real: primeras dos semanas.
sample = pred_df.head(24 * 14)

plt.figure(figsize=(14, 6))
plt.plot(sample["DATETIME"], sample["actual"], label="Real", linewidth=2)
plt.plot(sample["DATETIME"], sample["prediction"], label="Prediccion", linewidth=2)
plt.title("Predicciones vs valores reales - primeras dos semanas")
plt.xlabel("Fecha")
plt.ylabel("SIN Imputed")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/predictions_vs_actual.png", dpi=160)
plt.show()

# Histograma de errores.
plt.figure(figsize=(8, 6))
plt.hist(pred_df["error"], bins=40, color="#0f766e", edgecolor="white")
plt.title("Histograma de errores")
plt.xlabel("Error = prediccion - valor real")
plt.ylabel("Frecuencia")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/error_histogram.png", dpi=160)
plt.show()

# Mejor RMSE por conjunto de features.
best_by_set = results.groupby("feature_set")["RMSE"].min().sort_values()

plt.figure(figsize=(10, 5))
best_by_set.plot(kind="bar", color="#2563eb")
plt.title("Mejor RMSE por conjunto de features")
plt.ylabel("RMSE")
plt.xticks(rotation=25, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/rmse_by_feature_set.png", dpi=160)
plt.show()"""
    ),
    md("## 10. Grafico para explicar energia y comportamiento"),
    code(
        """# Consumo promedio por hora y por tipo de dia.
plot_df = processed.copy()
plot_idx = plot_df.index.tz_convert("America/Asuncion")
plot_df["hour"] = plot_idx.hour
plot_df["day_type"] = np.where(plot_idx.dayofweek >= 5, "Fin de semana", "Dia laboral")

hourly = plot_df.groupby(["hour", "day_type"])[TARGET].mean().reset_index()

plt.figure(figsize=(12, 5))
for label, group in hourly.groupby("day_type"):
    plt.plot(group["hour"], group[TARGET], marker="o", label=label)

plt.title("Promedio de energia por hora: dia laboral vs fin de semana")
plt.xlabel("Hora del dia")
plt.ylabel("Promedio de SIN Imputed")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/energy_by_hour_weekend.png", dpi=160)
plt.show()"""
    ),
    md("## 11. Descargar resultados"),
    code(
        """with zipfile.ZipFile("avance_proyecto_ia_colab.zip", "w") as z:
    for folder in ["data_processed", "results", "figures"]:
        for root, _, files_in_folder in os.walk(folder):
            for name in files_in_folder:
                path = os.path.join(root, name)
                z.write(path)

files.download("avance_proyecto_ia_colab.zip")"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

GUIDE.write_text(
    """# Guia rapida para correr el proyecto en Google Colab

1. Abrir Google Colab: https://colab.research.google.com/
2. Subir el notebook `notebooks/00_colab_ejecucion_completa.ipynb`.
3. Ejecutar la primera celda de carga de archivo.
4. Seleccionar `NuevoTPIA.csv` desde tu computadora.
5. Ir ejecutando las celdas en orden: Entorno de ejecucion > Ejecutar todo.
6. Al final se descarga `avance_proyecto_ia_colab.zip` con:
   - dataset procesado,
   - resultados de Grid Search,
   - predicciones,
   - metricas,
   - figuras.

Para mostrar al profesor, enfocate en:

- Las features nuevas: fin de semana, feriados, noche, manana, tarde, hora pico, sensibilidad termica y memoria corta.
- La tabla `hyperparameter_results.csv`.
- El resultado del mejor modelo.
- Los graficos de prediccion vs real y consumo promedio por hora.
""",
    encoding="utf-8",
)

print(f"Notebook creado: {OUT}")
print(f"Guia creada: {GUIDE}")
