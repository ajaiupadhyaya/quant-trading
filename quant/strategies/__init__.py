"""Strategy registry. Subclasses register themselves via @register."""

from __future__ import annotations

import importlib
import pkgutil

from quant.strategies.base import Strategy, StrategySpec

REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator that adds a Strategy subclass to the registry."""
    slug = cls.spec.slug
    if slug in REGISTRY:
        raise ValueError(f"Strategy slug {slug!r} is already registered")
    REGISTRY[slug] = cls
    return cls


def list_strategies() -> list[StrategySpec]:
    """Return the StrategySpecs for all registered strategies, sorted by slug."""
    return [REGISTRY[k].spec for k in sorted(REGISTRY)]


def _autoload_strategies() -> None:
    """Import every submodule of quant.strategies so @register decorators fire.

    Concrete strategy classes land in Plans 4 and 5; this hook means each new
    file becomes discoverable just by existing — no manual __init__ edits.
    """
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name == "base":
            continue
        importlib.import_module(f"{__name__}.{module_info.name}")


_autoload_strategies()


__all__ = ["REGISTRY", "Strategy", "StrategySpec", "list_strategies", "register"]
