"""ML server implementation using Flask. Provides endpoints for processing images with specified parameters."""

import base64
import io
import time
import uuid
from functools import lru_cache

from flask import Flask, jsonify, request
from PIL import Image

from contracts.contracts import ShadowParams


@lru_cache(maxsize=1)
def _get_process_image():
    # тяжёлые импорты будут только при первом вызове
    from ML_SERVER.processor import process_image

    return process_image


app = Flask(__name__)


@app.route("/test", methods=["GET"])
def test():
    """Simple test endpoint to check if the server is running."""
    return "OK"


@app.route("/process", methods=["POST"])
def process_compat():
    """Legacy endpoint for processing images. Redirects to the new /v1/process endpoint."""
    return process_v1()


@app.route("/v1/process", methods=["POST"])
def process_v1():
    """Main endpoint for processing images. Expects an image file and parameters in the request.
    Returns processed images and metadata."""
    request_id = str(uuid.uuid4())
    t0 = time.time()

    # 1) достаём картинку
    if "image" not in request.files:
        return jsonify(
            {
                "request_id": request_id,
                "error": {"code": "NO_IMAGE", "message": "Field 'image' is required"},
            }
        ), 400

    image_file = request.files["image"]
    try:
        image = Image.open(image_file.stream).convert("RGB")
    except Exception as e:
        return jsonify(
            {
                "request_id": request_id,
                "error": {"code": "BAD_IMAGE", "message": f"Cannot read image: {e}"},
            }
        ), 400

    params_obj = ShadowParams.from_form(request.form)
    params = params_obj.__dict__

    def to_int(v, default):
        try:
            return int(v)
        except Exception:
            return default

    params["rot"] = to_int(params.get("rot", 0), 0)

    # 3) запускаем твой текущий пайплайн
    t_proc0 = time.time()
    try:
        process_image = _get_process_image()
        processed_images, processed_text = process_image(image, params)
    except Exception as e:
        return jsonify(
            {
                "request_id": request_id,
                "error": {"code": "PROCESSING_FAILED", "message": str(e)},
            }
        ), 500
    t_proc_ms = int((time.time() - t_proc0) * 1000)

    # 4) пакуем картинки в base64
    out = []
    for pil_img in processed_images:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        out.append({"mime": "image/jpeg", "b64": b64})

    dt_ms = int((time.time() - t0) * 1000)

    return jsonify(
        {
            "api_version": "1.0",
            "request_id": request_id,
            "message": processed_text,
            "images": out,
            "meta": {"timings_ms": {"total": dt_ms, "processing": t_proc_ms}},
            "warnings": [],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9001)
