from collections import deque

from flask import Flask, request, render_template_string, redirect, url_for
import base64
from collections import deque

app = Flask(__name__)

# Хранилище для полученных изображений (в base64)
received_images = deque(maxlen=15)

# HTML-шаблон для отображения изображений
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Uploaded Images</title>
</head>
<body>
    <h1>Uploaded Images</h1>
    <div style="display: flex; flex-wrap: wrap; gap: 20px;">
        {% for img in images %}
            <div style="border: 1px solid #ccc; padding: 10px;">
                <img src="data:image/png;base64,{{ img }}" alt="Image" style="max-width: 300px; height: auto;">
            </div>
        {% endfor %}
    </div>
    <script>
        setTimeout(() => {
            location.reload();  // Автоматическое обновление страницы каждые 5 секунд
        }, 5000);
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def show_images():
    return render_template_string(HTML_TEMPLATE, images=received_images)

@app.route('/upload', methods=['POST'])
def upload_images():
    global received_images

    # Получаем изображения из POST-запроса
    files = request.files.getlist('images')
    for file in files:
        # Читаем файл и кодируем в base64
        image_data = file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        received_images.appendleft(encoded_image)

    # Перенаправляем обратно на главную страницу для отображения новых изображений
    return redirect(url_for('show_images'))

if __name__ == '__main__':
    app.run(debug=True, port=9002)
