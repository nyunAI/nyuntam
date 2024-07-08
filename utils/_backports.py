def _import_StrEnum():
    try:
        from enum import StrEnum
    except ImportError:
        from strenum import StrEnum

    return StrEnum

StrEnum = _import_StrEnum()

def __getattr__(name):
    if name == 'StrEnum':
        return StrEnum
    
    raise AttributeError(f"module {__name__} has no attribute {name}")