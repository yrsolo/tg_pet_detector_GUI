import requests
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import gradio as gr
import io

import gradio as gr
from PIL import Image
import random
import os
import sys



import random

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
    image.save(buffered, format="JPEG", quality=95)  # Сохраняем изображение в формате JPEG
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


    processed_images = []
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
            processed_images.append(Image.open(io.BytesIO(body.strip())))

    if processed_images is [] or processed_text is None:
        raise ValueError("Не удалось обработать ответ от сервера")

    return processed_images, processed_text

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

    processed_images, text = handle_server_response(response)

    if len(processed_images) < 4:
        processed_images += processed_images[:1]*(4-len(processed_images))

    if response.status_code == 200:
        return processed_images #, "Обработка завершена!" + text
    else:
        return [image]*4 #, f"Ошибка при обработке изображения: {response.status_code}"

# Функция для выбора крупного изображения
def select_image(index, images):
    index = int(index)
    return images[index][0]

# Интерфейс

with gr.Blocks() as app:
    gr.Markdown("# Object Detection App")
    gr.Markdown("Загрузите изображение для обработки. На выходе вы получите 4 варианта, которые можно просмотреть в большом формате.")

    # Загрузка изображения
    with gr.Row():
        image_input = gr.Image(type="pil", label="Загрузите изображение")
        # image_input = gr.Video(label="Снимите фото или загрузите видео", sources=["upload", "webcam"])


    with gr.Row():
        process_button = gr.Button("Обработать")

    # Отображение миниатюр и выбор изображения
    with gr.Row():
        thumbnails = gr.Gallery(label="Миниатюры", columns=2, rows=2)
    selected_index = gr.Number(label="Выберите индекс изображения для просмотра", interactive=True)
    display_image = gr.Image(label="Выбранное изображение")

    # Перезагрузка приложения
    reload_button = gr.Button("Перезагрузить приложение")

    # Связь между компонентами
    process_button.click(
        fn=process_image_server,
        inputs=image_input,
        outputs=thumbnails,
    )
    thumbnails.select(
        fn=select_image,
        inputs=[selected_index, thumbnails],
        outputs=display_image,
    )

    def reload_app():
        """Перезапуск приложения."""
        python = sys.executable
        os.execl(python, python, *sys.argv)
        pass

        # return

    reload_button.click(
        fn=reload_app,
        inputs=[],
        outputs=[],
    )


if __name__ == "__main__":
    if ENV == "production":
        print("Запуск в продакшен среде с SSL...")
        app.launch(
            server_name="0.0.0.0",  # Домен сервера
            ssl_verify=False,  # Отключаем проверку SSL
            server_port=7860,
            ssl_keyfile=PRIVATE_KEY_PATH,  # Приватный ключ
            ssl_certfile=CERTIFICATE_PATH,  # Сертификат
        )

    else:
        print("Запуск в локальной среде без SSL...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
        )

