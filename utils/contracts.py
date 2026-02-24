# contracts.py
from dataclasses import dataclass

@dataclass
class ShadowParams:
    rot: int = 0
    max_objects: int = 4
    return_debug: bool = False

    def to_dict(self) -> dict:
        """
        Готовит параметры к отправке через requests.post(..., data=...)
        (там всё должно быть строками)
        """
        return {
            "rot": str(self.rot),
            "max_objects": str(self.max_objects),
            "return_debug": "1" if self.return_debug else "0",
        }

    @staticmethod
    def from_form(form: dict) -> "ShadowParams":
        """
        Парсит request.form (там всё строки) -> нормальные типы
        """
        def to_int(v, default):
            try:
                return int(v)
            except Exception:
                return default

        def to_bool(v, default=False):
            if v is None:
                return default
            return str(v).lower() in ("1", "true", "yes", "y", "on")

        return ShadowParams(
            rot=to_int(form.get("rot"), 0),
            max_objects=to_int(form.get("max_objects"), 4),
            return_debug=to_bool(form.get("return_debug"), False),
        )