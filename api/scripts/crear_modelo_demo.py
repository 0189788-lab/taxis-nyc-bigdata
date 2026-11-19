"""Genera artefactos de modelo FALSOS (datos sintéticos) para probar la API
sin haber entrenado todavía el modelo real en Colab.

    python scripts/crear_modelo_demo.py            # escribe en ./modelo_demo
    python scripts/crear_modelo_demo.py ruta/dir   # o donde indiques

Luego:  MODEL_DIR=modelo_demo uvicorn main:app --reload
NO uses esto para la entrega: el modelo real sale de notebooks/02_modelo_ml.py.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

BOROUGHS = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island"]
FEATURES = ["PULocationID", "DOLocationID", "borough_origen_cod", "borough_destino_cod",
            "hora", "dia_num", "passenger_count", "dist_ruta", "dur_ruta", "log_n_ruta"]


def crear(destino: Path, n: int = 5000, seed: int = 0) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    ids = np.arange(1, 266)
    boro = rng.choice(BOROUGHS + ["Unknown"], size=len(ids))
    boro[ids == 1] = "EWR"
    boro[ids == 132] = "Queens"
    boro[ids == 230] = "Manhattan"
    boro[ids >= 264] = "Unknown"
    zonas = pd.DataFrame({"LocationID": ids, "Borough": boro, "Zone": [f"Zona {i}" for i in ids]})
    zonas.loc[zonas.LocationID == 132, "Zone"] = "JFK Airport"
    zonas.loc[zonas.LocationID == 230, "Zone"] = "Times Sq/Theatre District"
    zonas.to_csv(destino / "zonas.csv", index=False)

    pares = pd.DataFrame({"PULocationID": rng.integers(1, 264, 3000), "DOLocationID": rng.integers(1, 264, 3000)})
    pares = pd.concat([pares, pd.DataFrame({"PULocationID": [132], "DOLocationID": [230]})]).drop_duplicates(
        ["PULocationID", "DOLocationID"])
    pares["n_ruta"] = rng.integers(20, 5000, len(pares))
    pares["dist_ruta"] = rng.uniform(0.5, 20, len(pares)).round(2)
    pares["dur_ruta"] = (pares["dist_ruta"] * 3 + rng.uniform(2, 10, len(pares))).round(1)
    pares.to_csv(destino / "rutas.csv", index=False)

    mapa_boro = {b: i for i, b in enumerate(BOROUGHS)}
    X = pd.DataFrame({
        "PULocationID": rng.integers(1, 264, n).astype(float),
        "DOLocationID": rng.integers(1, 264, n).astype(float),
        "borough_origen_cod": rng.choice([*mapa_boro.values(), np.nan], n),
        "borough_destino_cod": rng.choice([*mapa_boro.values(), np.nan], n),
        "hora": rng.integers(0, 24, n).astype(float),
        "dia_num": rng.integers(1, 8, n).astype(float),
        "passenger_count": rng.integers(1, 7, n).astype(float),
        "dist_ruta": rng.uniform(0.5, 20, n),
        "dur_ruta": rng.uniform(3, 70, n),
        "log_n_ruta": np.log1p(rng.integers(20, 5000, n)),
    })[FEATURES]
    X.loc[rng.random(n) < 0.05, ["dist_ruta", "dur_ruta", "log_n_ruta"]] = np.nan
    trafico = 1 + 0.4 * ((X.hora >= 16) & (X.hora <= 19))
    y_tarifa = 3 + 2.8 * X.dist_ruta.fillna(3) + rng.normal(0, 2, n)
    y_dur = X.dur_ruta.fillna(15) * trafico + rng.normal(0, 3, n)

    cat_idx = [FEATURES.index("borough_origen_cod"), FEATURES.index("borough_destino_cod")]
    modelos = {}
    for nombre, y in [("tarifa", y_tarifa), ("duracion", y_dur)]:
        m = HistGradientBoostingRegressor(max_iter=50, categorical_features=cat_idx, random_state=seed)
        modelos[nombre] = m.fit(X, y)
    joblib.dump(modelos, destino / "modelos.joblib", compress=3)

    meta = {"version": "demo-sintetico", "features": FEATURES, "boroughs": BOROUGHS,
            "categoricas": ["borough_origen_cod", "borough_destino_cod"],
            "metricas_test": {"nota": "modelo de demostración con datos sintéticos"},
            "sklearn_version": sklearn.__version__}
    (destino / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destino


if __name__ == "__main__":
    salida = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("modelo_demo")
    print("Artefactos demo en", crear(salida).resolve())
