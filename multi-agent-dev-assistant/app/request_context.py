"""Request-scoped context used to annotate application logs."""

from contextvars import ContextVar
import logging


request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdLogFilter(logging.Filter):
    """Attach the current request ID to each formatted log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True
