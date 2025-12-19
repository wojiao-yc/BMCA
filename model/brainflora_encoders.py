import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict, List

_EXPORTS: List[str] = [
    "Cogcap",
    "NeV2L",
    "MindEyeModule",
    "NICE",
    "EEGNetv4_Encoder",
    "EEGConformer_Encoder",
    "EEGITNet_Encoder",
    "Projector",
    "ShallowFBCSPNet_Encoder",
    "ATCNet_Encoder",
    "MetaEEG",
]

_MODULE_CACHE: Dict[str, ModuleType] = {}
_MODULE_NAME = "_bmca_brainflora_contrast"
_CONTRAST_PATH = (
    Path(__file__).resolve().parents[2]
    / "BrainFLORA"
    / "Retrieval"
    / "contrast_retrieval.py"
)


def _load_module() -> ModuleType:
    """Lazy-load the BrainFLORA contrast retrieval module."""
    cache_key = str(_CONTRAST_PATH)
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    if not _CONTRAST_PATH.exists():
        raise FileNotFoundError(
            f"Expected BrainFLORA contrast_retrieval.py at {_CONTRAST_PATH}"
        )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _CONTRAST_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {_CONTRAST_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _MODULE_CACHE[cache_key] = module
    return module


def available_encoders() -> Dict[str, type]:
    """Return the subset of encoder classes exported from BrainFLORA."""
    module = _load_module()
    encoders: Dict[str, type] = {}
    for name in _EXPORTS:
        if hasattr(module, name):
            encoders[name] = getattr(module, name)
    return encoders


def __getattr__(name: str):
    """Expose exported encoders as module-level attributes."""
    if name in _EXPORTS:
        module = _load_module()
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"{name} is not available in brainflora_encoders")


__all__ = _EXPORTS
