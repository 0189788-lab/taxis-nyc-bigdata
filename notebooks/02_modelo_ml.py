# -*- coding: utf-8 -*-
"""
02_modelo_ml.py  —  Capa de Machine Learning (Taxis NYC)

Pégalo en Colab DESPUÉS de haber generado la capa Silver (notebook 01).
Cada bloque  # %%  es una celda.

Problema: REGRESIÓN. Dado origen, destino, hora, día y pasajeros,
estimar (1) la tarifa base (fare_amount, USD) y (2) la duración (minutos).
Métrica principal: MAE (error absoluto medio) — se interpreta directo en
dólares y minutos y es menos sensible a outliers que el RMSE.
También se reportan RMSE y R², y se compara contra un baseline
(la mediana histórica de la ruta).

Salida (en Drive):  Bigdata/modelo/
    modelos.joblib   -> dict {"tarifa": modelo, "duracion": modelo}
    rutas.csv        -> estadísticas por ruta (las usa la API como features)
    zonas.csv        -> catálogo de zonas
    metadata.json    -> versión, features, métricas, versión de sklearn
"""

# %% ---------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------------
# !pip install pyspark            # (si es una sesión nueva de Colab)
# from google.colab import drive
# drive.mount('/content/drive')

import json, os, sys, datetime
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.appName("taxis_nyc_ml")
         .config("spark.driver.memory", "8g")
         .getOrCreate())
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

ruta = "/content/drive/MyDrive/Bigdata/"
ruta_modelo = ruta + "modelo/"
os.makedirs(ruta_modelo, exist_ok=True)

SEED = 42
N_TRAIN = 1_500_000   # filas de entrenamiento que bajamos a pandas
N_TEST = 400_000      # filas de prueba

# Orden fijo de boroughs -> códigos (la API usa exactamente esta lista)
BOROUGHS = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island"]
FEATURES = ["PULocationID", "DOLocationID", "borough_origen_cod", "borough_destino_cod",
            "hora", "dia_num", "passenger_count", "dist_ruta", "dur_ruta", "log_n_ruta"]
CATEGORICAS = ["borough_origen_cod", "borough_destino_cod"]

# %% ---------------------------------------------------------------
# 1. Dataset de modelado (Spark, sobre TODA la capa Silver)
# ------------------------------------------------------------------
silver = spark.read.parquet(ruta + "silver/taxi_trips")

base = (silver
    # 264 y 265 son "Unknown" / "Outside of NYC": no sirven para cotizar
    .filter(F.col("PULocationID").between(1, 263) & F.col("DOLocationID").between(1, 263))
    # outliers extremos de tarifa (> $250 son <0.01% y casi siempre errores o viajes negociados)
    .filter(F.col("fare_amount") < 250)
    .filter(F.col("passenger_count").between(1, 6))
    .withColumn("dia_num", F.dayofweek("tpep_pickup_datetime"))   # 1=domingo ... 7=sábado
    .select("PULocationID", "DOLocationID", "borough_origen", "borough_destino",
            "hora", "dia_num", "mes",
            F.col("passenger_count").cast("int").alias("passenger_count"),
            "trip_distance", "fare_amount", "duracion_min"))

# Split TEMPORAL: entrenamos con junio-julio y probamos con agosto.
# Así medimos cómo se comporta el modelo con datos "del futuro" (sin fuga de información).
train_sp = base.filter(F.col("mes").isin(6, 7))
test_sp = base.filter(F.col("mes") == 8)

# %% ---------------------------------------------------------------
# 2. Features por ruta (agregación en Spark, SOLO con datos de entrenamiento)
# ------------------------------------------------------------------
rutas = (train_sp.groupBy("PULocationID", "DOLocationID")
    .agg(F.count("*").alias("n_ruta"),
         F.percentile_approx("trip_distance", 0.5).alias("dist_ruta"),
         F.percentile_approx("duracion_min", 0.5).alias("dur_ruta"),
         F.percentile_approx("fare_amount", 0.5).alias("tarifa_ruta"))   # solo para el baseline
    .filter(F.col("n_ruta") >= 20))   # rutas con muy pocos viajes no son confiables

print("Rutas con estadísticas:", rutas.count())

def con_rutas(df):
    return df.join(F.broadcast(rutas), on=["PULocationID", "DOLocationID"], how="left")

n_train_total = train_sp.count()
n_test_total = test_sp.count()
print("Viajes train (jun-jul):", n_train_total, "| test (ago):", n_test_total)

# Muestreo aleatorio para entrenar en memoria (1.5M filas es suficiente para este modelo)
frac_train = min(1.0, N_TRAIN / n_train_total)
frac_test = min(1.0, N_TEST / n_test_total)

train_pd = con_rutas(train_sp).sample(fraction=frac_train, seed=SEED).toPandas()
test_pd = con_rutas(test_sp).sample(fraction=frac_test, seed=SEED).toPandas()
print("Muestra train:", len(train_pd), "| muestra test:", len(test_pd))

# %% ---------------------------------------------------------------
# 3. Preparar matrices (MISMA lógica que usa la API en predictor.py)
# ------------------------------------------------------------------
def codificar_borough(serie):
    codes = pd.Categorical(serie, categories=BOROUGHS).codes.astype("float64")
    codes[codes < 0] = np.nan     # borough desconocido -> NaN (el modelo lo maneja)
    return codes

def preparar_X(df):
    X = pd.DataFrame({
        "PULocationID": df["PULocationID"].astype("float64"),
        "DOLocationID": df["DOLocationID"].astype("float64"),
        "borough_origen_cod": codificar_borough(df["borough_origen"]),
        "borough_destino_cod": codificar_borough(df["borough_destino"]),
        "hora": df["hora"].astype("float64"),
        "dia_num": df["dia_num"].astype("float64"),
        "passenger_count": df["passenger_count"].astype("float64"),
        "dist_ruta": df["dist_ruta"].astype("float64"),
        "dur_ruta": df["dur_ruta"].astype("float64"),
        "log_n_ruta": np.log1p(df["n_ruta"].astype("float64")),
    })
    return X[FEATURES]

X_train, X_test = preparar_X(train_pd), preparar_X(test_pd)
y_train = {"tarifa": train_pd["fare_amount"].values, "duracion": train_pd["duracion_min"].values}
y_test = {"tarifa": test_pd["fare_amount"].values, "duracion": test_pd["duracion_min"].values}

# %% ---------------------------------------------------------------
# 4. Entrenar (Gradient Boosting) + evaluar contra baseline
# ------------------------------------------------------------------
def nuevo_modelo():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.1,
        max_iter=400,
        max_leaf_nodes=63,
        min_samples_leaf=100,
        l2_regularization=1.0,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAS],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=SEED,
    )

def metricas(y, pred):
    return {"MAE": round(float(mean_absolute_error(y, pred)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
            "R2": round(float(r2_score(y, pred)), 4)}

modelos, resultados = {}, {}
baseline_col = {"tarifa": "tarifa_ruta", "duracion": "dur_ruta"}

for objetivo in ["tarifa", "duracion"]:
    m = nuevo_modelo()
    m.fit(X_train, y_train[objetivo])
    modelos[objetivo] = m

    pred = m.predict(X_test)
    # Baseline: mediana histórica de la ruta (o mediana global si la ruta no existe)
    base_pred = test_pd[baseline_col[objetivo]].fillna(np.median(y_train[objetivo])).values

    resultados[objetivo] = {"modelo": metricas(y_test[objetivo], pred),
                            "baseline_mediana_ruta": metricas(y_test[objetivo], base_pred),
                            "iteraciones": int(m.n_iter_)}
    print(objetivo.upper(), json.dumps(resultados[objetivo], indent=2))

# %% ---------------------------------------------------------------
# 5. Exportar artefactos para la API
# ------------------------------------------------------------------
version = datetime.datetime.now().strftime("%Y%m%d-%H%M")

joblib.dump(modelos, ruta_modelo + "modelos.joblib", compress=3)

(rutas.select("PULocationID", "DOLocationID", "n_ruta", "dist_ruta", "dur_ruta")
      .toPandas()
      .to_csv(ruta_modelo + "rutas.csv", index=False))

(spark.read.csv(ruta + "taxi_zone_lookup.csv", header=True, inferSchema=True)
      .select("LocationID", "Borough", "Zone")
      .toPandas()
      .to_csv(ruta_modelo + "zonas.csv", index=False))

metadata = {
    "version": version,
    "tarea": "regresion",
    "objetivos": {"tarifa": "fare_amount (USD)", "duracion": "duracion_min (minutos)"},
    "metrica_principal": "MAE",
    "features": FEATURES,
    "categoricas": CATEGORICAS,
    "boroughs": BOROUGHS,
    "dia_num": "1=domingo, 2=lunes, ..., 7=sabado (convención de Spark dayofweek)",
    "entrenamiento": {"periodo": "2026-06 a 2026-07", "filas_muestra": len(train_pd),
                      "filas_disponibles": int(n_train_total)},
    "prueba": {"periodo": "2026-08", "filas_muestra": len(test_pd),
               "filas_disponibles": int(n_test_total)},
    "metricas_test": resultados,
    "sklearn_version": sklearn.__version__,
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
    "python_version": sys.version.split()[0],
}
with open(ruta_modelo + "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Artefactos guardados en", ruta_modelo)
print(os.listdir(ruta_modelo))
print("\n>>> Copia estas líneas en api/requirements.txt:")
print(f"scikit-learn=={sklearn.__version__}")
print(f"pandas=={pd.__version__}")
print(f"numpy=={np.__version__}")
import sys
print(f"\n>>> Y en api/Dockerfile usa:  FROM python:{sys.version_info.major}.{sys.version_info.minor}-slim")

# %% ---------------------------------------------------------------
# 6. (Opcional) Prueba rápida del modelo cargado desde disco
# ------------------------------------------------------------------
cargados = joblib.load(ruta_modelo + "modelos.joblib")
ejemplo = X_test.head(5)
print(pd.DataFrame({
    "tarifa_real": y_test["tarifa"][:5], "tarifa_pred": cargados["tarifa"].predict(ejemplo).round(2),
    "dur_real": y_test["duracion"][:5].round(1), "dur_pred": cargados["duracion"].predict(ejemplo).round(1),
}))
