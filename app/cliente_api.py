"""Cliente HTTP para la API publicada del cotizador.

Toda la comunicación con el modelo pasa por aquí: la app NO tiene el modelo,
solo conoce la URL del endpoint. Maneja:
  * tiempos de espera (conexión y lectura por separado)
  * reintentos con backoff ante 502/503/504 y fallos de conexión
  * errores HTTP (401, 422 con detalle por campo, 4xx, 5xx)
  * respuestas inválidas (no-JSON, campos faltantes, valores fuera de rango)
"""
from __future__ import annotations

import math
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ErrorAPI(Exception):
    """Error con un mensaje listo para mostrar al usuario y un tipo para la lógica de la app."""

    def __init__(self, mensaje: str, tipo: str, status: int | None = None):
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.tipo = tipo          # timeout | conexion | no_disponible | auth | validacion | cliente | servidor | respuesta_invalida
        self.status = status


CAMPOS_PREDICCION = {
    "tarifa_estimada_usd": (0, 1000),
    "duracion_estimada_min": (0, 600),
}


class ClienteAPI:
    def __init__(self, base_url: str, api_key: str | None = None,
                 timeout_conexion: float = 5, timeout_lectura: float = 30, reintentos: int = 2):
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ErrorAPI("La URL de la API no está configurada o no es válida.", "configuracion")
        self.base_url = base_url.rstrip("/")
        self.timeout = (timeout_conexion, timeout_lectura)

        retry = Retry(
            total=reintentos,
            connect=reintentos,
            read=False,                  # si el servidor ya recibió la petición y tarda, no la repetimos a ciegas
            status=reintentos,
            backoff_factor=1.0,          # espera 0s, 2s, 4s ... entre intentos
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),   # /predict es idempotente
            raise_on_status=False,       # devolvemos la última respuesta y la interpretamos nosotros
        )
        self.session = requests.Session()
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"Accept": "application/json"})
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    # ------------------------------------------------------------------
    def _request(self, metodo: str, ruta: str, **kwargs) -> Any:
        url = f"{self.base_url}{ruta}"
        try:
            r = self.session.request(metodo, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectTimeout:
            raise ErrorAPI(f"No se pudo conectar con la API en {self.timeout[0]:.0f}s.", "timeout")
        except requests.exceptions.ReadTimeout:
            raise ErrorAPI(
                f"La API no respondió en {self.timeout[1]:.0f}s. Si el servidor estaba dormido "
                "(plan gratuito), espera un minuto y vuelve a intentar.", "timeout")
        except requests.exceptions.SSLError:
            raise ErrorAPI("Error de certificado SSL al conectar con la API.", "conexion")
        except requests.exceptions.ConnectionError:
            raise ErrorAPI("No hay conexión con la API (¿URL correcta? ¿servicio encendido?).", "conexion")
        except requests.exceptions.RequestException as e:
            raise ErrorAPI(f"Error inesperado de red: {e.__class__.__name__}", "conexion")

        if r.status_code >= 400:
            self._lanzar_error_http(r)

        try:
            return r.json()
        except ValueError:
            raise ErrorAPI("La API respondió algo que no es JSON válido.", "respuesta_invalida", r.status_code)

    @staticmethod
    def _lanzar_error_http(r: requests.Response):
        try:
            detalle = r.json().get("detail")
        except (ValueError, AttributeError):
            detalle = None

        s = r.status_code
        if s in (401, 403):
            raise ErrorAPI("La API rechazó la credencial (X-API-Key).", "auth", s)
        if s == 422:
            if isinstance(detalle, list):   # formato de FastAPI: [{"loc": [...], "msg": "..."}]
                partes = [f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detalle]
                detalle = "; ".join(partes)
            raise ErrorAPI(f"Datos inválidos — {detalle or 'revisa los campos'}", "validacion", s)
        if s in (502, 503, 504):
            raise ErrorAPI(f"La API no está disponible por ahora ({s}). {detalle or ''}".strip(), "no_disponible", s)
        if s >= 500:
            raise ErrorAPI(f"Error interno de la API ({s}).", "servidor", s)
        raise ErrorAPI(f"La API rechazó la petición ({s}): {detalle or r.reason}", "cliente", s)

    # ------------------------------------------------------------------
    def salud(self) -> dict:
        data = self._request("GET", "/health")
        if not isinstance(data, dict) or "estado" not in data:
            raise ErrorAPI("Respuesta de /health con formato inesperado.", "respuesta_invalida")
        return data

    def zonas(self) -> list[dict]:
        data = self._request("GET", "/zonas")
        if not isinstance(data, list) or not data or not all(
                isinstance(z, dict) and {"id", "zona", "borough"} <= z.keys() for z in data):
            raise ErrorAPI("La lista de zonas llegó vacía o con formato inesperado.", "respuesta_invalida")
        return data

    def predecir(self, origen_id: int, destino_id: int, hora: int, dia_semana: str, pasajeros: int) -> dict:
        payload = {"origen_id": int(origen_id), "destino_id": int(destino_id), "hora": int(hora),
                   "dia_semana": dia_semana, "pasajeros": int(pasajeros)}
        data = self._request("POST", "/predict", json=payload)
        return self.validar_prediccion(data)

    @staticmethod
    def validar_prediccion(data: Any) -> dict:
        """No confiamos ciegamente en la respuesta: verificamos tipos y rangos razonables."""
        if not isinstance(data, dict):
            raise ErrorAPI("La predicción no tiene el formato esperado.", "respuesta_invalida")
        for campo, (minimo, maximo) in CAMPOS_PREDICCION.items():
            valor = data.get(campo)
            if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
                raise ErrorAPI(f"La API devolvió un valor inválido en '{campo}'.", "respuesta_invalida")
            if not (minimo <= valor <= maximo):
                raise ErrorAPI(f"'{campo}'={valor} está fuera del rango razonable ({minimo}–{maximo}).",
                               "respuesta_invalida")
        return data
