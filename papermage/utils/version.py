import importlib.metadata
from pathlib import Path


def get_version() -> str:
    """Get the version of the package."""
    # This is a workaround for the fact that if the package is installed
    # in editable mode, the version is not reliability available.
    # Therefore, we check for the existence of a file called EDITABLE,
    # which is not included in the package at distribution time.
    path = Path(__file__).parent / "EDITABLE"
    if path.exists():
        return "dev"

    try:
        # package has been installed, so it has a version number
        # from pyproject.toml
        version = importlib.metadata.version(get_name())
    except importlib.metadata.PackageNotFoundError:
        # package hasn't been installed, so set version to "dev"
        version = "dev"

    return version


def get_name() -> str:
    """Get the name of the package."""
    return "smashed"


def get_name_and_version() -> str:
    """Get the name and version of the package."""
    return f"{get_name()}=={get_version()}"
