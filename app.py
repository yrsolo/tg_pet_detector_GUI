import requests
from PIL import Image
import gradio as gr

SERVER_URL = "http://127.0.0.1:5000/process"

# Заглушка: эмуляция обработки на сервере
def process_image(image):
    # Преобразуем изображение в байты для отправки
    image_bytes = image.tobytes()

    # Отправляем изображение на сервер
    response = requests.post(SERVER_URL, files={"image": image_bytes})

    if response.status_code == 200:
        # Получаем обработанный результат (здесь возвращается текст)
        message = response.json().get("message", "Ошибка на сервере")
        processed_image = response.json().get("image", None)
        return processed_image, message
    else:
        return image, "Ошибка при обработке изображения!"


# Интерфейс
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="Object Detection App",
    description="Загрузите изображение, чтобы обработать его",
)

if __name__ == "__main__":
    interface.launch()
