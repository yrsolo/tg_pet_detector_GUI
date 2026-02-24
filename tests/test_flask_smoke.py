def test_flask_test_endpoint():
    import ML_server

    client = ML_server.app.test_client()
    r = client.get("/test")
    assert r.status_code == 200
    assert r.data.decode("utf-8") == "OK"
