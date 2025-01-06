import torch

import typing

import numpy as np
import requests
from PIL import Image
import io

from constant import device, MAX_MAMO_SIZE

def pic2int(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = image.transpose((1, 2, 0))

    if isinstance(image, Image.Image):
        image = np.array(image)

    pic_max = image.max()
    pic_min = image.min()

    if pic_min < 0 or pic_max > 255:
        e = 1e-8
        image = (image - pic_min + e) / (pic_max - pic_min + e)
        pic_max = image.max()

    if pic_max <= 1:
        image = image * 255

    return image.astype('uint8')


def pic2float(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = image.transpose((1, 2, 0))

    if isinstance(image, Image.Image):
        image = np.array(image)

    pic_max = image.max()
    pic_min = image.min()

    if pic_min < 0:
        e = 1e-8
        image = (image - pic_min + e) / (pic_max - pic_min + e)
    elif pic_max > 1:
        image = image / 255
    else:
        image = image.astype('float32')

    return image


def pic2pil(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = image.transpose((1, 2, 0))
    if isinstance(image, np.ndarray):
        image = pic2int(image)
        image = Image.fromarray(image)
    return image


def swimg(image_arrays, server_url="http://127.0.0.1:9002/upload"):
    """
    Отправляет список numpy массивов на сервер Flask.

    :param image_arrays: Список numpy массивов (изображений)
    :param server_url: URL сервера Flask
    """

    image_arrays = [pic2int(i) for i in image_arrays]
    files = []
    # print(image_arrays)

    for idx, array in enumerate(image_arrays):
        # Преобразуем numpy массив в изображение (если нужно, приводим к uint8)
        # if array.dtype != np.uint8:
        #    array = (array * 255).astype(np.uint8)

        # Если изображение черно-белое, добавляем канал
        if len(array.shape) == 2:  # Grayscale (H, W)
            array = np.stack([array] * 3, axis=-1)  # Convert to (H, W, 3)

        # Преобразуем массив в изображение с помощью PIL
        # print(array.shape, array.min(), array.max(), array.dtype)
        image = Image.fromarray(array)

        # Сохраняем изображение в буфер памяти
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Добавляем изображение в список файлов
        files.append(('images', (f'image_{idx}.png', buffer, 'image/png')))

    # Отправляем POST-запрос с изображениями
    response = requests.post(server_url, files=files)

    # Проверяем статус
    # if response.status_code == 200:
    #     print(f"Successfully sent {len(image_arrays)} images to the server!")
    # else:
    #     print(f"Failed to send images! Server response: {response.text}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def hash_image(image):
    image = pic2int(image)
    image = Image.fromarray(image)
    image = image.resize((512, 512))
    image = image.convert("RGB")
    image = image.tobytes()
    return hash(image)


def hash_all(x):
    if isinstance(x, typing.Union[list, tuple]):
        return hash(sum(hash_all(i) for i in x))

    if isinstance(x, dict):
        return hash(sum(hash_all(v) for v in x.values()))

    if isinstance(x, np.ndarray):
        return hash(x.tobytes())

    if isinstance(x, torch.Tensor):
        return hash(x.cpu().detach().numpy().tobytes())

    if isinstance(x, Image.Image):
        return hash_image(x)

    if isinstance(x, typing.Hashable):
        # print(type(x))
        return hash(x)

    return hash(str(x))


def memo(func):
    m = {}
    def wrapper(*arg, **kwarg):
        if len(m) > MAX_MAMO_SIZE:
            m.clear()
        hash = hash_all((*arg, *kwarg.items()))
        if hash not in m:
            m[hash] = func(*arg, **kwarg)
        return m[hash]


    return wrapper