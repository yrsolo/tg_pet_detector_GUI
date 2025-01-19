from ML_SERVER.sam import sam_process
from ML_SERVER.blip import describe_center_object
from ML_SERVER.gdino import detect_objects
from utils import pic2pil, pic2float
from PIL import Image
from SHADOW.pix2pix import generate_shadow
from utils import memo
from functools import lru_cache

@memo
def process_image(image, params):
    """
    Обработка изображения и текста.

    :param image: объект PIL.Image, изображение
    :param text: str, текст
    :return: tuple, объект PIL.Image и текст
    """
    image = pic2float(image)

    # blip
    description = describe_center_object(image)

    # gdino
    objects = detect_objects(image, description)
    bbox = objects['boxes'][0].tolist()

    # sam
    processed_images, mask, text = sam_process(image, bbox=bbox)

    if 'rot' in params:
        rot = int(params['rot'])
    else:
        rot = None

    processed_images = generate_shadow(processed_images, mask, rots=rot)

    processed_text = str(text) + ' good'

    processed_images = [pic2pil(img) for img in processed_images]

    return processed_images, processed_text

def test():
    image = Image.open("..\image.jpg")
    text = 'test'
    processed_images, processed_text = process_image(image, text)
    print(processed_text)

if __name__ == '__main__':
    test()