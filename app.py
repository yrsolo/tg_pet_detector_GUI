import requests
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import gradio as gr
import io

from typing import Dict

import gradio as gr
from PIL import Image
import random
import os
import sys

# from utils import memo

import random

import base64
import io
from PIL import Image

from WEB.ml_api import MLClient, MLResponse
from utils.contracts import ShadowParams

# SERVER_URL = "http://127.0.0.1:9001/process"
SERVER_URL = "http://127.0.0.1:9001/"
# Путь к вашим сертификатам (только для сервера)
CERTIFICATE_PATH = '/opt/shadowgen/https-cert/certificate.pem'
PRIVATE_KEY_PATH = '/opt/shadowgen/https-cert/private_key.pem'

# Получаем окружение из переменных окружения
ENV = os.getenv("ENV", "local")  # По умолчанию "local"

# @memo
def process_image_server(image, rot, max_size=1024, max_pic=2):
    # Преобразуем изображение в JPEG-формат и отправляем
    params = {
        'rot': rot
    }

    if image.size[0] > max_size or image.size[1] > max_size:
        image.thumbnail((max_size, max_size))


    client = MLClient(SERVER_URL)
    params = ShadowParams(rot=rot, max_objects=4, return_debug=False)

    try:
        response = client.process(image, params)
    except Exception as e:        
        print(f"Ошибка при отправке запроса: {e}")        

    return response.images 


# Функция для выбора крупного изображения
def select_image(index, images):
    index = int(index)
    return images[index][0]

# Интерфейс

with gr.Blocks() as app:
    gr.Markdown("# Shadow Generator")
    gr.Markdown("Загрузите изображение для обработки. \nКнопки +/- 20 вращают тень")

    # Загрузка изображения
    with gr.Row():
        image_input = gr.Image(type="pil", label="Загрузите изображение")

    with gr.Row():
        # Обработать
        process_button = gr.Button("Обработать")

        # Блок для ввода угла
        # with gr.Row():
        angle_input = gr.Number(value=0, label="Угол тени (0-359)", visible=False)#, interactive=True)

        # Стрелочки для изменения значения
        # with gr.Row():
        decrease_button = gr.Button("-20")
        increase_button = gr.Button("+20")

        # Галочка для автоматической перегенерации
        # auto_generate = gr.Checkbox(label="Перегенерировать автоматически при изменении угла")

    # Отображение миниатюр и выбор изображения
    with gr.Row():
        thumbnails = gr.Gallery(label="Миниатюры", columns=2, rows=2)
    # selected_index = gr.Number(label="Выберите индекс изображения для просмотра", interactive=True)
    # display_image = gr.Image(label="Выбранное изображение")

    # Перезагрузка приложения
    reload_button = gr.Button("Перезагрузить приложение")


    def update_angle(current_angle, delta, image_input):
        # Изменение угла на указанное значение (±20)

        new_angle = (current_angle + delta) % 360
        angle_input.value = new_angle

        processed_images = process_image(image_input, new_angle)

        return new_angle, processed_images

    def process_image(image, rot):
        processed_images = process_image_server(image, rot)
        return processed_images

    def reload_app():
        """Перезапуск приложения."""
        python = sys.executable
        os.execl(python, python, *sys.argv)

    decrease_button.click(
        fn=update_angle,
        inputs=[angle_input, gr.State(-20), image_input],
        outputs=[angle_input,thumbnails],
    )

    increase_button.click(
        fn=update_angle,
        inputs=[angle_input, gr.State(20), image_input],
        outputs=[angle_input,thumbnails]
    )

    process_button.click(
        fn=process_image,
        inputs=[image_input, angle_input],
        outputs=thumbnails,
    )

    reload_button.click(
        fn=reload_app,
        inputs=[],
        outputs=[],
    )

    # thumbnails.select(
    #     fn=lambda idx, thumbs: thumbs[idx],
    #     inputs=[selected_index, thumbnails],
    #     outputs=display_image,
    # )


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

