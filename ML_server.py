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

app = Flask(__name__)


def prepare_response(processed_images, processed_text, status=200):
    """
    Подготавливает ответ для возвращения из функции process.
    Возвращает изображение и текст в формате multipart/form-data.

    :param processed_image: объект PIL.Image, обработанное изображение
    :param processed_text: str, обработанный текст
    :return: Response объект Flask
    """
    # Сохраняем изображение в буфер
    # buffer = io.BytesIO()
    # processed_image.save(buffer, format="JPEG")
    # buffer.seek(0)

    # Формируем multipart-ответ
    boundary = "----CustomBoundaryString"
    response_body = []

    # Добавляем текст
    response_body.append(f"--{boundary}")
    response_body.append('Content-Disposition: form-data; name="message"')
    response_body.append("")
    response_body.append(processed_text)

    # Добавляем изображение
    for i, image in enumerate(processed_images):
        response_body.append(f"--{boundary}")
        response_body.append(f'Content-Disposition: form-data; name="image"; filename="processed_image_{i}.jpg"')
        response_body.append("Content-Type: image/jpeg")
        response_body.append("")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        response_body.append(buffer.getvalue())
    # response_body.append('TEST TEST TEST')

    # Закрываем boundary
    response_body.append(f"--{boundary}--")
    # response_body.append("")

    # Создаём HTTP-ответ
    response_body = b"\r\n".join(
        part if isinstance(part, bytes) else part.encode("utf-8") for part in response_body
    )
    # print(response_body)
    response = Response(
        response_body,
        content_type=f"multipart/form-data; boundary={boundary}",
        status=status,
    )

    return response

@app.route('/test', methods=['GET'])
def test():
    return 'OK'

# @app.route('/process', methods=['POST'])
# def process():
#     try:
#         # Получаем изображение из запроса
#         if 'image' not in request.files:
#             return prepare_response(None, 'Изображение не найдено', 400)

#         print(request.form)

#         image_file = request.files['image']
#         image = Image.open(image_file)

#         params = request.form

#         # Здесь может быть ML-обработка
#         # Например, обработка изображения (в данном случае просто возвращаем обратно)
#         processed_images, text = process_image(image, params)
#         # print(f'processed_image with shape {processed_image.size}')


#         return prepare_response(processed_images, text, 200)
#     except Exception as e:
#         # Логируем ошибку (пока print, потом заменим на logging)
#         print(f"Ошибка обработки: {e}")
#         return prepare_response(None, f'Ошибка обработки: {str(e)}', 500)

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
    params = dict(request.form)  # безопасно: как раньше
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
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        out.append({"mime": "image/png", "b64": b64})

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
