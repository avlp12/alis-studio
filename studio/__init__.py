"""Alis Studio — a local, model-agnostic image-generation studio for Apple silicon (MLX)."""

# Single source of truth for the version. pyproject.toml reads this via
# [tool.setuptools.dynamic], the server injects it into the web UI, and the
# native window/DMG build stamp it into the bundle — keep it here only.
__version__ = "0.6.2"
