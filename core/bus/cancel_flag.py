import threading

_tls = threading.local()


class GenerationCancelled(Exception):
    pass


def set_cancel_event(event: threading.Event | None) -> None:
    _tls.event = event


def get_cancel_event() -> threading.Event | None:
    return getattr(_tls, "event", None)


def check_cancelled() -> None:
    event = get_cancel_event()
    if event is not None and event.is_set():
        raise GenerationCancelled()
