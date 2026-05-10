"""``lens_xt`` — PEP-8 import alias for the ``lensx`` package.

Both spellings work and resolve to the same module:

    from lens_xt import LensX     # PEP-8 form (matches `lens-xt` distribution name)
    from lensx import LensX       # short form (matches CLI binary name)

This shim re-exports everything from ``lensx`` so any future additions to
the public API stay in sync without code changes here.
"""
from __future__ import annotations

import lensx as _lensx
from lensx import *  # noqa: F401, F403  -- re-export every public name

# Mirror metadata for tools that introspect __version__ via the alias name
__version__ = _lensx.__version__
__all__ = list(_lensx.__all__)
