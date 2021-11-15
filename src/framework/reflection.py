import importlib


def load_class(module_name: str, class_name: str) -> any:
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)
