# contracts.py
from dataclasses import dataclass
from typing import Any


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
    def from_dict(form: dict) -> "ShadowParams":
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


@dataclass
class TimingsMs:
    total: float
    processing: float

    @staticmethod
    def from_json(data: dict[str, Any]) -> "TimingsMs":
        # допускаем int/float/str, приводим к float
        def to_float(v: Any) -> float:
            try:
                return float(v)
            except Exception:
                raise ValueError(f"Invalid timing value: {v!r}")

        return TimingsMs(
            total=to_float(data.get("total", 0)),
            processing=to_float(data.get("processing", 0)),
        )


@dataclass
class ProcessMeta:
    timings_ms: TimingsMs

    @staticmethod
    def from_json(data: dict[str, Any]) -> "ProcessMeta":
        timings = data.get("timings_ms", {})
        return ProcessMeta(timings_ms=TimingsMs.from_json(timings))


@dataclass
class ProcessResponse:
    api_version: str
    request_id: str
    message: str
    images: Any  # пока не типизируем жёстко, т.к. out может быть списком/объектом
    meta: ProcessMeta
    warnings: list[Any]

    @staticmethod
    def from_json(data: dict[str, Any]) -> "ProcessResponse":
        for key in ("api_version", "request_id", "message", "images", "meta", "warnings"):
            if key not in data:
                raise ValueError(f"Missing '{key}' in response")

        return ProcessResponse(
            api_version=str(data["api_version"]),
            request_id=str(data["request_id"]),
            message=str(data["message"]),
            images=data["images"],
            meta=ProcessMeta.from_json(data["meta"]),
            warnings=list(data["warnings"]),
        )
