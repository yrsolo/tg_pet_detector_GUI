from flask import Flask, request, jsonify, send_file, Response
import io
from PIL import Image
from ML_SERVER.sam import sam_process, pic2pil, pic2float
from ML_SERVER.processor import process_image

import base64
import io
import uuid
import time
from PIL import Image
from flask import Flask, request, jsonify

from utils.contracts import ShadowParams

app = Flask(__name__)


@app.route('/test', methods=['GET'])
def test():
    return 'OK'

@app.route("/process", methods=["POST"])
def process_compat():
    return process_v1()

@app.route("/v1/process", methods=["POST"])
def process_v1():
    request_id = str(uuid.uuid4())
    t0 = time.time()

    # 1) достаём картинку
    if "image" not in request.files:
        return jsonify({
            "request_id": request_id,
            "error": {"code": "NO_IMAGE", "message": "Field 'image' is required"}
        }), 400

    image_file = request.files["image"]
    try:
        image = Image.open(image_file.stream).convert("RGB")
    except Exception as e:
        return jsonify({
            "request_id": request_id,
            "error": {"code": "BAD_IMAGE", "message": f"Cannot read image: {e}"}
        }), 400

    # 2) параметры (можешь оставить как есть: rot из form)
    #    но лучше: params_json в поле params
    # params = dict(request.form)  # безопасно: как раньше
    params_obj = ShadowParams.from_form(request.form)
    params = params_obj.__dict__ 
    # если хочешь: params["rot"] -> int(params.get("rot", 0)) и т.п.

    def to_int(v, default):
        try:
            return int(v)
        except Exception:
            return default
        
    params["rot"] = to_int(params.get("rot", 0), 0)

    # 3) запускаем твой текущий пайплайн
    try:
        processed_images, processed_text = process_image(image, params)
    except Exception as e:
        return jsonify({
            "request_id": request_id,
            "error": {"code": "PROCESSING_FAILED", "message": str(e)}
        }), 500

    # 4) пакуем картинки в base64
    out = []
    for pil_img in processed_images:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        out.append({"mime": "image/jpeg", "b64": b64})

    dt_ms = int((time.time() - t0) * 1000)

    return jsonify({
        "api_version": "1.0",
        "request_id": request_id,
        "message": processed_text,
        "images": out,
        "meta": {"timings_ms": {"total": dt_ms}},
        "warnings": []
    })


if __name__ == "__main__":
    pass
    app.run(debug=True, host="0.0.0.0", port=9001)
