import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog

_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_job_id: ContextVar[Optional[str]] = ContextVar("job_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(rid: Optional[str]) -> None:
    _request_id.set(rid)


def set_job_id(jid: Optional[str]) -> None:
    _job_id.set(jid)


def get_context() -> Dict[str, Any]:
    return {
        "request_id": _request_id.get(),
        "job_id": _job_id.get(),
    }


def bind_context(**kwargs: Any) -> None:
    if "request_id" in kwargs:
        set_request_id(kwargs["request_id"])
    if "job_id" in kwargs:
        set_job_id(kwargs["job_id"])


def _add_context(_: Any, __: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Примешиваем request_id/job_id в каждую запись
    ctx = get_context()
    for k, v in ctx.items():
        if v is not None and k not in event_dict:
            event_dict[k] = v
    return event_dict


def _add_service(_: Any, __: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    event_dict.setdefault("service", os.getenv("SHADOWGEN_SERVICE", "unknown"))
    event_dict.setdefault("env", os.getenv("SHADOWGEN_ENV", "dev"))
    return event_dict


def setup_logging(level: Optional[str] = None) -> None:
    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, lvl, logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_service,
            _add_context,
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,  # stacktrace в поле exception
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "shadowgen"):
    return structlog.get_logger(name)


class log_timing:
    """Контекст-менеджер для замеров duration_ms."""

    def __init__(self, logger, event: str, **fields: Any):
        self.logger = logger
        self.event = event
        self.fields = fields
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.logger.info(self.event + "_start", **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb):
        dur_ms = (time.perf_counter() - self.t0) * 1000
        if exc:
            self.logger.error(
                self.event + "_error", duration_ms=round(dur_ms, 2), exc_info=True, **self.fields
            )
            return False
        self.logger.info(self.event + "_done", duration_ms=round(dur_ms, 2), **self.fields)
        return False
