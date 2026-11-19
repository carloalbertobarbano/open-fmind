"""open-fmind: parcellated fMRI timeseries from OpenNeuro, one subject at a time."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("open-fmind")
except PackageNotFoundError:
    __version__ = "0.0.0"
