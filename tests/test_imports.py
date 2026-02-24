import importlib


def test_import_app_module():
    importlib.import_module("app")


def test_import_ml_server_module():
    importlib.import_module("ML_server")
