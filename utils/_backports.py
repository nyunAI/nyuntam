def _import_StrEnum():
    try:
        from enum import StrEnum
    except ImportError:
        # TODO: remove this block after strenum is added to dockers.
        try:
            from strenum import StrEnum
        except ModuleNotFoundError:
            # installs strenum
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "strenum"])
            return _import_StrEnum()
    return StrEnum


StrEnum = _import_StrEnum()


def __getattr__(name):
    if name == "StrEnum":
        return StrEnum

    raise AttributeError(f"module {__name__} has no attribute {name}")
