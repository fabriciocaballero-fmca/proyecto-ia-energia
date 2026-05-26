import { Presentation, PresentationFile } from "file:///C:/Users/fabri/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
import { readFileSync } from "node:fs";

const root = "C:/Users/fabri/Documents/Codex/2026-05-25/files-mentioned-by-the-user-nbeatsnuevo";
const summaryRaw = JSON.parse(readFileSync(`${root}/results/metrics_summary.json`, "utf8"));
const summary = {
  RMSE: summaryRaw.RMSE,
  MAE: summaryRaw.MAE,
  MAPE: summaryRaw["MAPE_%"],
  R2: summaryRaw.R2,
};

const slides = [
  ["Pronostico horario de SIN", ["Benjamin", "Proyecto de Inteligencia Artificial", "Series temporales y variables meteorologicas"], "predictions_vs_actual.png"],
  ["Planteamiento del problema", ["Predecir el comportamiento horario de SIN", "Apoyar analisis energetico y planificacion", "Integrar clima y patrones historicos"], null],
  ["Dataset", ["Archivo NuevoTPIA.csv", "Frecuencia horaria", "Objetivo: SIN Imputed", "Covariables: temperatura, humedad, presion, lluvia y viento"], null],
  ["Features creados", ["Ciclos calendario: hora, dia y mes", "Fin de semana, feriados y franjas horarias", "Rezagos cortos y semanales", "Sensibilidad termica y medias moviles", "Magnitud de viento e interaccion clima"], null],
  ["Modelo utilizado", ["Baseline: regresion Ridge", "Regularizacion para reducir sobreajuste", "Notebook original: N-BEATS como alternativa profunda"], null],
  ["Optimizacion", ["Grid Search manual", "Alphas: 0 a 1000", "Comparacion de tres conjuntos de features"], "grid_search_curve.png"],
  ["Visualizacion de resultados", ["Curva de busqueda de hiperparametros", "Predicciones vs valores reales", "Histograma de errores"], "rmse_by_feature_set.png"],
  ["Predicciones y evaluacion", [`RMSE: ${summary.RMSE.toFixed(2)}`, `MAE: ${summary.MAE.toFixed(2)}`, `MAPE: ${summary.MAPE.toFixed(2)}%`, `R2: ${summary.R2.toFixed(4)}`], "error_histogram.png"],
  ["Conclusiones y limitaciones", ["Los rezagos capturan estacionalidad fuerte", "El modelo lineal es interpretable y reproducible", "Limitacion: no captura todas las no linealidades"], null],
  ["Trabajos futuros", ["Optimizar N-BEATS con Optuna", "Comparar con boosting y modelos recurrentes", "Publicar GitHub Pages y completar datos personales"], null],
];

const presentation = new Presentation(null, {
  id: "proyecto-ia-sin",
  slides: [],
  theme: {
    colors: {
      accent1: "#0F766E",
      accent2: "#2563EB",
      accent3: "#DC2626",
      background1: "#F8FAFC",
      text1: "#0F172A",
    },
  },
});

for (const [title, bullets, image] of slides) {
  const slide = presentation.slides.add();
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: 914, height: 514 },
    fill: { type: "solid", color: "#F8FAFC" },
    line: { style: "none" },
  }).sendToBack();
  const titleShape = slide.shapes.add({
    geometry: "rect",
    position: { left: 42, top: 34, width: 820, height: 78 },
    fill: { type: "solid", color: "#F8FAFC" },
    line: { style: "none" },
  });
  titleShape.text.style = { fontSize: 28, bold: true, typeface: "Arial", color: "#0F172A" };
  titleShape.text = title;

  const body = slide.shapes.add({
    geometry: "rect",
    position: { left: 58, top: 130, width: image ? 430 : 780, height: 320 },
    fill: { type: "solid", color: "#F8FAFC" },
    line: { style: "none" },
  });
  body.text.style = { fontSize: 18, typeface: "Arial", color: "#172033" };
  body.text = bullets.map((b) => `• ${b}`).join("\n");

  slide.shapes.add({
    geometry: "rect",
    position: { left: 42, top: 474, width: 820, height: 6 },
    fill: { type: "solid", color: "#0F766E" },
    line: { style: "none" },
  });

  if (image) {
    slide.images.add({
      path: `${root}/assets/${image}`,
      alt: title,
      position: { left: 515, top: 132, width: 360, height: 270 },
      fit: "contain",
    });
  }
}

const blob = await PresentationFile.exportPptx(presentation);
await blob.save(`${root}/presentation/presentacion_proyecto_ia.pptx`);
