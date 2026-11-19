"""Pruebas del manejo de errores del cliente contra un servidor falso local.

    pip install pytest requests
    pytest -q app/tests
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cliente_api import ClienteAPI, ErrorAPI  # noqa: E402

PRED_OK = {"tarifa_estimada_usd": 52.3, "duracion_estimada_min": 38.0, "distancia_ruta_millas": 16.9,
           "ruta_conocida": True, "viajes_historicos_ruta": 1200,
           "origen": {"id": 132, "zona": "JFK Airport", "borough": "Queens"},
           "destino": {"id": 230, "zona": "Times Sq", "borough": "Manhattan"}, "modelo_version": "t"}

ESTADO = {"modo": "ok", "llamadas": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _enviar(self, status, cuerpo, tipo="application/json"):
        datos = cuerpo if isinstance(cuerpo, bytes) else json.dumps(cuerpo).encode()
        self.send_response(status)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        ESTADO["llamadas"] += 1
        if self.path == "/health":
            self._enviar(200, {"estado": "ok", "modelo_version": "t"})
        elif self.path == "/zonas":
            self._enviar(200, [{"id": 1, "zona": "Newark", "borough": "EWR"}])

    def do_POST(self):
        ESTADO["llamadas"] += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        modo = ESTADO["modo"]
        if modo == "ok":
            self._enviar(200, PRED_OK)
        elif modo == "lento":
            time.sleep(2)
            self._enviar(200, PRED_OK)
        elif modo == "422":
            self._enviar(422, {"detail": [{"loc": ["body", "hora"], "msg": "Input should be <= 23"}]})
        elif modo == "401":
            self._enviar(401, {"detail": "API key inválida"})
        elif modo == "500":
            self._enviar(500, {"detail": "Error interno del servidor"})
        elif modo == "503":
            self._enviar(503, {"detail": "El modelo no está cargado"})
        elif modo == "503_luego_ok":
            if ESTADO["llamadas"] == 1:
                self._enviar(503, {"detail": "despertando"})
            else:
                self._enviar(200, PRED_OK)
        elif modo == "html":
            self._enviar(200, b"<html>Bad gateway</html>", "text/html")
        elif modo == "sin_campos":
            self._enviar(200, {"hola": 1})
        elif modo == "negativo":
            self._enviar(200, {**PRED_OK, "tarifa_estimada_usd": -5})
        elif modo == "texto":
            self._enviar(200, {**PRED_OK, "duracion_estimada_min": "veinte"})


@pytest.fixture(scope="module")
def url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def cliente(url, **kw):
    kw.setdefault("timeout_lectura", 1)
    kw.setdefault("reintentos", 1)
    return ClienteAPI(url, **kw)


def predecir(c):
    return c.predecir(132, 230, 18, "viernes", 1)


def modo(m):
    ESTADO["modo"], ESTADO["llamadas"] = m, 0


def test_ok(url):
    modo("ok")
    assert predecir(cliente(url))["tarifa_estimada_usd"] == 52.3
    assert cliente(url).salud()["estado"] == "ok"
    assert cliente(url).zonas()[0]["id"] == 1


@pytest.mark.parametrize("m,tipo", [
    ("lento", "timeout"), ("422", "validacion"), ("401", "auth"), ("500", "servidor"),
    ("503", "no_disponible"), ("html", "respuesta_invalida"), ("sin_campos", "respuesta_invalida"),
    ("negativo", "respuesta_invalida"), ("texto", "respuesta_invalida"),
])
def test_errores(url, m, tipo):
    modo(m)
    with pytest.raises(ErrorAPI) as e:
        predecir(cliente(url, reintentos=0))
    assert e.value.tipo == tipo, e.value.mensaje


def test_422_mensaje_legible(url):
    modo("422")
    with pytest.raises(ErrorAPI) as e:
        predecir(cliente(url))
    assert "hora" in e.value.mensaje and "23" in e.value.mensaje


def test_reintenta_503(url):
    modo("503_luego_ok")
    assert predecir(cliente(url, reintentos=2))["duracion_estimada_min"] == 38.0
    assert ESTADO["llamadas"] == 2


def test_servidor_apagado():
    with pytest.raises(ErrorAPI) as e:
        predecir(ClienteAPI("http://127.0.0.1:9", reintentos=0, timeout_conexion=1))
    assert e.value.tipo in ("conexion", "timeout")


def test_url_invalida():
    with pytest.raises(ErrorAPI):
        ClienteAPI("")
