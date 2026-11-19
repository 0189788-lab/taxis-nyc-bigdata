"""Pruebas de la API con un modelo sintético.

    pip install -r requirements-dev.txt
    pytest -q
"""
import importlib
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from crear_modelo_demo import crear
    import os
    os.environ["MODEL_DIR"] = str(crear(tmp_path_factory.mktemp("modelo")))
    os.environ.pop("API_KEY", None)
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:   # el "with" ejecuta el lifespan (carga del modelo)
        yield c


VALIDO = {"origen_id": 132, "destino_id": 230, "hora": 18, "dia_semana": "viernes", "pasajeros": 1}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["estado"] == "ok"


def test_swagger_disponible(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_zonas(client):
    zonas = client.get("/zonas").json()
    assert len(zonas) == 263 and {"id", "zona", "borough"} <= zonas[0].keys()


def test_predict_ok(client):
    r = client.post("/predict", json=VALIDO)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tarifa_estimada_usd"] >= 3
    assert body["duracion_estimada_min"] >= 1
    assert body["ruta_conocida"] is True
    assert body["origen"]["zona"] == "JFK Airport"


def test_dia_con_acento_y_mayusculas(client):
    assert client.post("/predict", json={**VALIDO, "dia_semana": "Sábado"}).status_code == 200


@pytest.mark.parametrize("cambio", [
    {"origen_id": 0}, {"origen_id": 264}, {"destino_id": "abc"}, {"hora": 24}, {"hora": -1},
    {"dia_semana": "feriado"}, {"pasajeros": 0}, {"pasajeros": 9}, {"campo_extra": 1},
])
def test_entradas_invalidas(client, cambio):
    assert client.post("/predict", json={**VALIDO, **cambio}).status_code == 422


def test_falta_campo(client):
    datos = dict(VALIDO)
    del datos["hora"]
    assert client.post("/predict", json=datos).status_code == 422


def test_lote(client):
    r = client.post("/predict/lote", json={"viajes": [VALIDO, {**VALIDO, "hora": 3}]})
    assert r.status_code == 200 and len(r.json()["predicciones"]) == 2


def test_lote_vacio(client):
    assert client.post("/predict/lote", json={"viajes": []}).status_code == 422
