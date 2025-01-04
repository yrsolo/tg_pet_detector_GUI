from flask import Flask, request, jsonify, send_file, Response
import io
from PIL import Image

app = Flask(__name__)


def prepare_response(processed_image, processed_text):
    """
    Подготавливает ответ для возвращения из функции process.
    Возвращает изображение и текст в формате multipart/form-data.

    :param processed_image: объект PIL.Image, обработанное изображение
    :param processed_text: str, обработанный текст
    :return: Response объект Flask
    """
    # Сохраняем изображение в буфер
    buffer = io.BytesIO()
    processed_image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Формируем multipart-ответ
    boundary = "----CustomBoundaryString"
    response_body = []

    # Добавляем текст
    response_body.append(f"--{boundary}")
    response_body.append('Content-Disposition: form-data; name="message"')
    response_body.append("")
    response_body.append(processed_text)

    # Добавляем изображение
    response_body.append(f"--{boundary}")
    response_body.append('Content-Disposition: form-data; name="image"; filename="processed_image.jpg"')
    response_body.append("Content-Type: image/jpeg")
    response_body.append("")
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
        status=200,
    )

    return response

def process_image(image, text):
    """
    Обработка изображения и текста.

    :param image: объект PIL.Image, изображение
    :param text: str, текст
    :return: tuple, объект PIL.Image и текст
    """
    # Здесь может быть ML-обработка
    # Например, обработка изображения (в данном случае просто возвращаем обратно)
    processed_image = image
    processed_text = 'good'

    return processed_image, processed_text

@app.route('/test', methods=['GET'])
def test():
    return 'OK'

@app.route('/process', methods=['POST'])
def process():
    # Получаем изображение из запроса
    if 'image' not in request.files:
        return jsonify({"error": "Изображение не найдено"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)

    if 'text' not in request.form:
        return jsonify({"error": "Текст не найден"}), 400

    params = request.form['text']

    # Здесь может быть ML-обработка
    # Например, обработка изображения (в данном случае просто возвращаем обратно)
    processed_image, text = process_image(image, params)
    print(f'processed_image with shape {processed_image.size}')


    return prepare_response(processed_image, text)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9001)
