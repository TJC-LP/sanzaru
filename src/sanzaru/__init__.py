from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("sanzaru")
except PackageNotFoundError:
    __version__ = "unknown"
