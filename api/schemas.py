"""Esquemas Pydantic: validan la entrada y documentan la API en Swagger/OpenAPI."""
from __future__ import annotations

import unicodedata
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DiaSemana = Literal["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# Convención de Spark dayofweek usada en el entrenamiento: 1=domingo ... 7=sábado
DIA_A_NUM = {"domingo": 1, "lunes": 2, "martes": 3, "miercoles": 4,
             "jueves": 5, "viernes": 6, "sabado": 7}


def _normalizar(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_acentos.strip().lower()


class SolicitudViaje(BaseModel):
    """Datos del viaje a cotizar."""
    model_config = ConfigDict(
        extra="forbid",   # rechaza campos que no existen (evita errores de dedo silenciosos)
        json_schema_extra={"examples": [
            {"origen_id": 132, "destino_id": 230, "hora": 18, "dia_semana": "viernes", "pasajeros": 1}
        ]},
    )

    origen_id: int = Field(..., ge=1, le=263, description="LocationID de la zona de origen (1–263). Ver GET /zonas")
    destino_id: int = Field(..., ge=1, le=263, description="LocationID de la zona de destino (1–263)")
    hora: int = Field(..., ge=0, le=23, description="Hora de salida (0–23)")
    dia_semana: DiaSemana = Field(..., description="Día de la semana (acepta acentos y mayúsculas)")
    pasajeros: int = Field(1, ge=1, le=6, description="Número de pasajeros (1–6)")

    @field_validator("dia_semana", mode="before")
    @classmethod
    def normalizar_dia(cls, v):
        return _normalizar(v) if isinstance(v, str) else v

    def a_dict_modelo(self) -> dict:
        return {"origen_id": self.origen_id, "destino_id": self.destino_id, "hora": self.hora,
                "dia_num": DIA_A_NUM[self.dia_semana], "pasajeros": self.pasajeros}


class SolicitudLote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    viajes: list[SolicitudViaje] = Field(..., min_length=1, max_length=100,
                                         description="Entre 1 y 100 viajes")


class Zona(BaseModel):
    id: int
    zona: str
    borough: str


class Prediccion(BaseModel):
    tarifa_estimada_usd: float = Field(..., description="Tarifa base estimada (fare_amount, sin propina ni peajes)")
    duracion_estimada_min: float
    distancia_ruta_millas: Optional[float] = Field(None, description="Distancia mediana histórica de la ruta")
    ruta_conocida: bool = Field(..., description="False si la ruta tuvo < 20 viajes en el entrenamiento (menos confiable)")
    viajes_historicos_ruta: int
    origen: Zona
    destino: Zona
    modelo_version: str


class RespuestaLote(BaseModel):
    predicciones: list[Prediccion]


class Salud(BaseModel):
    estado: Literal["ok", "sin_modelo"]
    modelo_version: Optional[str] = None
    metricas_test: Optional[dict] = None


class ErrorRespuesta(BaseModel):
    detail: str
