import importlib
import time


def test_ml_server_import_is_fast():
    t0 = time.time()
    importlib.import_module("ML_server")
    dt = time.time() - t0

    # Подстрой порог под свой ПК; обычно после lazy-import это < 1 сек.
    assert dt < 2.0, (
        f"ML_server импортируется слишком долго: {dt:.2f}s (возможно, грузится модель/torch)"
    )
