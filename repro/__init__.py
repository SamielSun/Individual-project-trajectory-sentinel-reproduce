"""Reproduction harness for the agent-trajectory-sentinel paper (arXiv:2608.02464).

This package is a thin, auditable overlay: it never vendors the upstream code.
`vendor/upstream` holds a pristine clone at a pinned commit; every script here
runs the upstream modules unmodified except where a `repro.mods` patch is
explicitly applied for the controlled modification.
"""

__version__ = "1.0.0"
