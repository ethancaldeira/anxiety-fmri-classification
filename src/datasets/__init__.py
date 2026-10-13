"""Dataset adapters: GSP (anxiety, STAI-T) and ABIDE (autism, DX_GROUP).

Pick an adapter by name:
    >>> from datasets import get_adapter
    >>> adapter = get_adapter('gsp')      # or 'abide'
"""

from .base import DatasetAdapter
from .gsp import GSPAdapter
from .abide import ABIDEAdapter


_ADAPTERS = {
    'gsp':   GSPAdapter,
    'abide': ABIDEAdapter,
}


def get_adapter(name: str, config=None) -> DatasetAdapter:
    """Return an instantiated dataset adapter, e.g. get_adapter('gsp', Config)."""
    key = name.lower()
    if key not in _ADAPTERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[key](config)
