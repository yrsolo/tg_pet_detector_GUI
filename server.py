from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    # Получаем изображение из запроса
    if 'image' not in request.files:
        return jsonify({"error": "Изображение не найдено"}), 400

    image = request.files['image']
    # Здесь может быть ML-обработка
    processed_image = image
    #return jsonify({"message": f"Изображение {image.filename} обработано успешно!"}), 200
    return jsonify({
        "message": f"Изображение обработано успешно!",
        "image": processed_image#.filename
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
