"""API REST del cotizador de taxis NYC.

Ejecutar local:   uvicorn main:app --reload
Documentación:    http://localhost:8000/docs   (Swagger)  |  /redoc  |  /openapi.json
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader

from predictor import Predictor
from schemas import (ErrorRespuesta, Prediccion, RespuestaLote, Salud,
                     SolicitudLote, SolicitudViaje, Zona)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

MODEL_DIR = os.getenv("MODEL_DIR", str(Path(__file__).parent / "modelo"))
API_KEY = os.getenv("API_KEY")   # opcional: si se define, /predict exige el header X-API-Key

estado: dict = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        estado["predictor"] = Predictor(MODEL_DIR)
    except Exception:
        # La API arranca igual para que /health reporte el problema en vez de morir sin explicación
        log.exception("No se pudo cargar el modelo desde %s", MODEL_DIR)
    yield


app = FastAPI(
    title="API Taxis NYC – Cotizador de viajes",
    version="1.0.0",
    description=(
        "Estima la **tarifa base** y la **duración** de un viaje en taxi amarillo de NYC "
        "a partir del origen, destino, hora, día y número de pasajeros.\n\n"
        "Modelo: Gradient Boosting entrenado con viajes TLC de junio–julio 2026 "
        "(capa Silver de la arquitectura Medallion) y evaluado con agosto 2026.\n\n"
        "Flujo típico: `GET /zonas` → elegir IDs → `POST /predict`."
    ),
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

ERRORES = {
    401: {"model": ErrorRespuesta, "description": "API key inválida o ausente"},
    422: {"description": "Entrada inválida (detalle por campo)"},
    503: {"model": ErrorRespuesta, "description": "Modelo no disponible"},
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar_api_key(key: str | None = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida o ausente (header X-API-Key)")


def obtener_predictor() -> Predictor:
    p = estado["predictor"]
    if p is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Revisa los logs del servidor.")
    return p


def validar_zonas(p: Predictor, s: SolicitudViaje, prefijo: str = ""):
    for campo in ("origen_id", "destino_id"):
        valor = getattr(s, campo)
        if not p.zona_existe(valor):
            raise HTTPException(status_code=422, detail=f"{prefijo}{campo}={valor} no es una zona válida")


# ----------------------------------------------------------------------
@app.middleware("http")
async def medir_tiempo(request: Request, call_next):
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    ms = (time.perf_counter() - inicio) * 1000
    respuesta.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, respuesta.status_code, ms)
    return respuesta


@app.exception_handler(Exception)
async def error_no_controlado(request: Request, exc: Exception):
    log.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})


# ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def raiz():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=Salud, tags=["Sistema"], summary="Estado del servicio y del modelo")
def health():
    p = estado["predictor"]
    if p is None:
        return JSONResponse(status_code=503, content={"estado": "sin_modelo"})
    return Salud(estado="ok", modelo_version=p.version, metricas_test=p.metadata.get("metricas_test"))


@app.get("/zonas", response_model=list[Zona], tags=["Catálogos"], responses={503: ERRORES[503]},
         summary="Lista de zonas válidas (id, nombre, borough)")
def zonas(p: Predictor = Depends(obtener_predictor)):
    return [Zona(id=z.id, zona=z.zona, borough=z.borough) for z in sorted(p.zonas.values(), key=lambda z: z.id)]


@app.post("/predict", response_model=Prediccion, tags=["Predicción"], responses=ERRORES,
          dependencies=[Depends(verificar_api_key)], summary="Cotiza un viaje")
def predict(solicitud: SolicitudViaje, p: Predictor = Depends(obtener_predictor)):
    validar_zonas(p, solicitud)
    return p.predecir([solicitud.a_dict_modelo()])[0]


@app.post("/predict/lote", response_model=RespuestaLote, tags=["Predicción"], responses=ERRORES,
          dependencies=[Depends(verificar_api_key)], summary="Cotiza hasta 100 viajes en una llamada")
def predict_lote(lote: SolicitudLote, p: Predictor = Depends(obtener_predictor)):
    for i, s in enumerate(lote.viajes):
        validar_zonas(p, s, prefijo=f"viajes[{i}].")
    return {"predicciones": p.predecir([s.a_dict_modelo() for s in lote.viajes])}
