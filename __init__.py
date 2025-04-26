# ComfyUI-DiaTest/__init__.py

# Import the final mappings from nodes.py
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Expose them for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### Loading: ComfyUI-DiaTest Nodes ###")