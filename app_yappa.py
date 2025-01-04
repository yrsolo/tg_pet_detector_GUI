from yappa import Yappa
from PIL import Image
import io

app = Yappa()

@app.route("/", methods=["GET"])
def index():
    return {"message": "Сервер работает!"}

@app.route("/process", methods=["POST"])
def process_image(request):
    # Получаем изображение из запроса
    file = request.files.get("image")
    if not file:
        return {"error": "Файл не найден"}, 400

    # Открываем изображение через PIL
    image = Image.open(io.BytesIO(file.read()))
    # Пример обработки: просто возвращаем размер изображения
    width, height = image.size
    return {"message": f"Обработка завершена! Размер изображения: {width}x{height}"}

if __name__ == "__main__":
    app.run(port=5000)
