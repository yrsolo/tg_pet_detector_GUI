import io

from PIL import Image

from contracts.contracts import ProcessResponse


def test_process_returns_400_without_image():
    import ML_server

    client = ML_server.app.test_client()
    r = client.post("/process", data={})
    assert r.status_code == 400


def test_process_happy_path_with_mock(monkeypatch):
    import ML_server

    # подменяем тяжёлую функцию обработчика
    def fake_process_image(image, params):
        return [image], "done"

    # если ты сделал lazy _get_process_image()
    monkeypatch.setattr(ML_server, "_get_process_image", lambda: fake_process_image)

    img = Image.new("RGB", (16, 16), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    client = ML_server.app.test_client()
    r = client.post(
        "/process",
        data={"image": (buf, "x.png"), "rot": "0"},
        content_type="multipart/form-data",
    )

    assert r.status_code == 200
    assert "application/json" in r.headers.get("Content-Type", "")

    data = r.get_json()
    assert data is not None

    resp = ProcessResponse.from_json(data)

    assert resp.api_version == "1.0"
    assert resp.request_id  # не пустой
    assert isinstance(resp.warnings, list)
    assert resp.meta.timings_ms.total >= 0
    assert resp.meta.timings_ms.processing >= 0


def test_process_returns_400_on_bad_image(monkeypatch):
    import ML_server

    # мок не нужен: пусть упадёт до ML
    client = ML_server.app.test_client()

    bad = io.BytesIO(b"not an image")
    r = client.post(
        "/process",
        data={"image": (bad, "x.png"), "rot": "0"},
        content_type="multipart/form-data",
    )

    assert r.status_code == 400
    assert "application/json" in r.headers.get("Content-Type", "")
    data = r.get_json()
    assert data is not None
    # проверяем, что вернулась ошибка, а не пустота
    assert "error" in data or "message" in data
