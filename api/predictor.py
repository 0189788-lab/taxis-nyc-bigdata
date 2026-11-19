"""Carga los artefactos del modelo y genera predicciones.

Construye las features EXACTAMENTE igual que notebooks/02_modelo_ml.py
(preparar_X), usando la lista de features y boroughs guardada en metadata.json.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger("predictor")

TARIFA_MINIMA = 3.0     # banderazo de NYC
DURACION_MINIMA = 1.0


@dataclass
class InfoZona:
    id: int
    zona: str
    borough: str


class Predictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        with open(model_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.version: str = self.metadata["version"]
        self.features: list[str] = self.metadata["features"]
        self.boroughs: list[str] = self.metadata["boroughs"]
        self.modelos = joblib.load(model_dir / "modelos.joblib")
        for clave in ("tarifa", "duracion"):
            if clave not in self.modelos:
                raise ValueError(f"modelos.joblib no contiene el modelo '{clave}'")

        zonas = pd.read_csv(model_dir / "zonas.csv")
        self.zonas: dict[int, InfoZona] = {
            int(r.LocationID): InfoZona(int(r.LocationID), str(r.Zone), str(r.Borough))
            for r in zonas.itertuples()
            if 1 <= int(r.LocationID) <= 263
        }

        rutas = pd.read_csv(model_dir / "rutas.csv")
        self.rutas: dict[tuple[int, int], tuple[float, float, int]] = {
            (int(r.PULocationID), int(r.DOLocationID)): (float(r.dist_ruta), float(r.dur_ruta), int(r.n_ruta))
            for r in rutas.itertuples()
        }
        log.info("Modelo %s cargado: %d zonas, %d rutas", self.version, len(self.zonas), len(self.rutas))

    # ------------------------------------------------------------------
    def zona_existe(self, zona_id: int) -> bool:
        return zona_id in self.zonas

    def _codigo_borough(self, borough: str) -> float:
        return float(self.boroughs.index(borough)) if borough in self.boroughs else np.nan

    def _fila(self, s: dict) -> dict:
        o, d = s["origen_id"], s["destino_id"]
        dist, dur, n = self.rutas.get((o, d), (np.nan, np.nan, 0))
        return {
            "PULocationID": float(o),
            "DOLocationID": float(d),
            "borough_origen_cod": self._codigo_borough(self.zonas[o].borough),
            "borough_destino_cod": self._codigo_borough(self.zonas[d].borough),
            "hora": float(s["hora"]),
            "dia_num": float(s["dia_num"]),
            "passenger_count": float(s["pasajeros"]),
            "dist_ruta": dist,
            "dur_ruta": dur,
            "log_n_ruta": float(np.log1p(n)) if n else np.nan,
        }

    def predecir(self, solicitudes: list[dict]) -> list[dict]:
        """solicitudes: dicts con origen_id, destino_id, hora, dia_num (1=dom..7=sáb), pasajeros."""
        X = pd.DataFrame([self._fila(s) for s in solicitudes])[self.features]
        tarifas = self.modelos["tarifa"].predict(X)
        duraciones = self.modelos["duracion"].predict(X)

        resultados = []
        for s, t, du in zip(solicitudes, tarifas, duraciones):
            if not (math.isfinite(t) and math.isfinite(du)):
                raise ValueError("El modelo devolvió un valor no numérico")
            o, d = s["origen_id"], s["destino_id"]
            dist, _, n = self.rutas.get((o, d), (None, None, 0))
            resultados.append({
                "tarifa_estimada_usd": round(max(float(t), TARIFA_MINIMA), 2),
                "duracion_estimada_min": round(max(float(du), DURACION_MINIMA), 1),
                "distancia_ruta_millas": round(dist, 2) if dist is not None else None,
                "ruta_conocida": n > 0,
                "viajes_historicos_ruta": int(n),
                "origen": asdict(self.zonas[o]),
                "destino": asdict(self.zonas[d]),
                "modelo_version": self.version,
            })
        return resultados
