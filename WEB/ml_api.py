#ml_api.py
import base64, io, requests
from dataclasses import dataclass
from PIL import Image
from utils.contracts import ShadowParams

@dataclass
class MLResponse:
    images: list[Image.Image]
    message: str
    meta: dict

class MLClient:
    def __init__(self, base_url: str, timeout=(5, 300)):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def process(self, pil_image: Image.Image, params: ShadowParams) -> MLResponse:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        resp = requests.post(
            f"{self.base_url}/v1/process",
            data=params.to_dict(),
            files={"image": ("image.jpg", buf.getvalue(), "image/jpeg")},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            e = data["error"]
            raise RuntimeError(f'{e.get("code","ERROR")}: {e.get("message","")}')

        images = []
        for item in data.get("images", []):
            img_bytes = base64.b64decode(item["b64"])
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        return MLResponse(images=images, message=data.get("message",""), meta=data.get("meta", {}))