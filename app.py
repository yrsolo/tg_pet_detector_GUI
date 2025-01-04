import requests
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import gradio as gr
import io

SERVER_URL = "http://127.0.0.1:9001/process"
# Путь к вашим сертификатам (только для сервера)
CERTIFICATE_PATH = '/home/yrsolo/tg-det/https-cert/certificate.pem'
PRIVATE_KEY_PATH = '/home/yrsolo/tg-det/https-cert/private_key.pem'

# Получаем окружение из переменных окружения
ENV = os.getenv("ENV", "local")  # По умолчанию "local"


def prepare_data(image: Image.Image, text: str):
    """
    Подготавливает данные для отправки через multipart/form-data.

    :param image: Объект изображения PIL.Image
    :param text: Строка текста
    :return: Словарь с подготовленными данными
    """
    # Преобразуем изображение в байты (JPEG)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)  # Сохраняем изображение в формате JPEG
    image_bytes = buffered.getvalue()  # Получаем байты изображения

    # Подготавливаем данные для отправки
    data = {
        "text": text,  # Текстовые данные
    }
    files = {
        "image": ("image.jpg", image_bytes, "image/jpeg")  # Картинка в формате (имя файла, содержимое, MIME-тип)
    }

    return data, files

def handle_server_response(response):
    """
    Обрабатывает multipart/form-data ответ от сервера.

    :param response: объект requests.Response, полученный от сервера
    :return: tuple (PIL.Image, str) — обработанное изображение и текст
    """
    # Проверяем успешность запроса
    if response.status_code != 200:
        raise ValueError(f"Ошибка запроса: {response.status_code} {response.reason}")

    # Получаем boundary из заголовка
    content_type = response.headers['Content-Type']
    boundary = content_type.split("boundary=")[-1]

    # print(response.content)
    # Разбиваем тело ответа по boundary
    parts = response.content.split(f"--{boundary}".encode())


    processed_image = None
    processed_text = None

    for part in parts:
        if not part.strip() or part == b"--":
            continue

        # Отделяем заголовки от тела
        # print('part - >  ', part[:200])
        headers, body = part.split(b"\r\n\r\n", 1)
        headers = headers.decode("utf-8")

        # Если это текст
        if 'name="message"' in headers:
            processed_text = body.decode("utf-8").strip()

        # Если это изображение
        elif 'name="image"' in headers:
            processed_image = Image.open(io.BytesIO(body.strip()))

    if processed_image is None or processed_text is None:
        raise ValueError("Не удалось обработать ответ от сервера")

    return processed_image, processed_text

# Заглушка: эмуляция обработки на сервере
def process_image(image):
    return image, "Обработка завершена!"

def process_image_server(image):
    # Преобразуем изображение в JPEG-формат и отправляем
    params = ''
    data, files = prepare_data(image, params)

    # Отправляем сжатый JPEG на сервер
    print(f'send to process image with size {image.size}')
    response = requests.post(
        SERVER_URL,
        data=data,
        files=files
    )

    print(response)

    processed_image, text = handle_server_response(response)
    if response.status_code == 200:
        return processed_image, "Обработка завершена!" + text
    else:
        return image, f"Ошибка при обработке изображения: {response.status_code}"


# Интерфейс
gradio_app = gr.Interface(
    fn=process_image_server,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="Object Detection App",
    description="Загрузите изображение для обработки."
)

if __name__ == "__main__":
    if ENV == "production":
        print("Запуск в продакшен среде с SSL...")
        gradio_app.launch(
            server_name="0.0.0.0",  # Домен сервера
            ssl_verify=False,  # Отключаем проверку SSL
            server_port=7860,
            ssl_keyfile=PRIVATE_KEY_PATH,  # Приватный ключ
            ssl_certfile=CERTIFICATE_PATH,  # Сертификат
        )

    else:
        print("Запуск в локальной среде без SSL...")
        gradio_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
        )

