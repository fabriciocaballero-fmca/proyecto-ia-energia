# Pronostico horario de SIN con variables meteorologicas

Proyecto de inteligencia artificial para estimar la serie horaria `SIN Imputed` a partir de variables meteorologicas y features temporales.

## Estructura

- `data/raw/`: dataset original.
- `data/processed/`: dataset limpio con features generados.
- `notebooks/`: preprocesamiento, entrenamiento, optimizacion, visualizacion y notebook N-BEATS original.
- `results/`: resultados CSV de Grid Search, predicciones y metricas.
- `assets/`: figuras usadas en reporte, presentacion y GitHub Pages.
- `docs/`: reporte en formato articulo IEEE y pagina `index.html`.
- `presentation/`: presentacion de 10 diapositivas.

## Features generados

| Grupo | Features | Como se generan | Por que ayudan |
|---|---|---|---|
| Calendario ciclico | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos` | Se transforma hora, dia y mes con seno/coseno. | Representan ciclos diarios, semanales y anuales sin cortes bruscos. |
| Comportamiento energetico | `is_weekend`, `is_night`, `is_morning`, `is_afternoon`, `is_peak_hour` | Se crean indicadores 0/1 segun dia y franja horaria. | Capturan habitos de consumo, como diferencias entre noche, tarde, hora pico y fin de semana. |
| Feriados | `is_holiday`, `is_day_before_holiday`, `is_day_after_holiday` | Se marcan feriados nacionales de Paraguay y dias cercanos. | Un feriado puede parecerse mas a un domingo que a un dia laboral normal. |
| Sensibilidad termica | `cooling_degree_24c`, `heating_degree_18c`, `hot_afternoon`, `hot_peak_hour` | Se mide cuanto la temperatura supera 24 C o baja de 18 C, y se combina con tarde/hora pico. | Aproxima el efecto de refrigeracion/calefaccion sobre la demanda. |
| Memoria temporal | `lag_1h`, `lag_2h`, `lag_3h`, `lag_24h`, `lag_168h`, `lag_336h` | Se usan valores anteriores de la serie. | La demanda horaria suele parecerse a horas recientes y al mismo horario de dias/semanas previas. |
| Promedios y extremos recientes | `rolling_3h_mean`, `rolling_6h_mean`, `rolling_12h_mean`, `rolling_24h_mean`, `temp_max_24h`, `target_max_24h` | Se calculan medias, maximos y minimos de ventanas anteriores. | Suavizan ruido y resumen el estado reciente del sistema. |
| Clima derivado | `wind_speed_10m`, `temp_humidity_interaction` | Se calcula magnitud del viento y producto temperatura-humedad. | Resume condiciones meteorologicas que pueden afectar el consumo. |

## Modelo y metricas

Se implemento un baseline de regresion Ridge con Grid Search manual sobre `alpha` y varios conjuntos de features. El mejor resultado fue:

- Feature set: `full_engineered`
- Alpha: `0.0`
- RMSE: `129.93`
- MAE: `91.49`
- MAPE: `4.66%`
- R2: `0.9438`

Como el problema es de regresion de una serie temporal, se usan RMSE, MAE, MAPE y R2. No se usa precision, F1-Score ni matriz de confusion porque esas metricas corresponden a problemas de clasificacion.

## Evaluacion preliminar

Se entreno con datos anteriores a 2021 y se valido desde 2021 en adelante. Esta separacion evita mezclar informacion futura en el entrenamiento. La tabla `results/hyperparameter_results.csv` permite comparar modelos y configuraciones, mientras que `results/predictions.csv` contiene los valores reales, predichos y errores.

## Visualizaciones

- `assets/predictions_vs_actual.png`: compara valores reales y predicciones.
- `assets/error_histogram.png`: muestra la distribucion de errores.
- `assets/rmse_by_feature_set.png`: compara el RMSE entre grupos de variables.
- `assets/grid_search_curve.png`: muestra el efecto del hiperparametro `alpha` en el RMSE.

El notebook `notebooks/nbeats_original.ipynb` conserva el enfoque avanzado con N-BEATS.
