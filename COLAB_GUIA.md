# Guia rapida para correr el proyecto en Google Colab

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
