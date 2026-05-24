"""Strategy registry. Subclasses register themselves via @register."""

from __future__ import annotations

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


__all__ = ["REGISTRY", "Strategy", "StrategySpec", "list_strategies", "register"]
