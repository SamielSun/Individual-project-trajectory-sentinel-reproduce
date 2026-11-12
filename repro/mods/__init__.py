"""Controlled modifications applied to the upstream monitors at run time.

Nothing here edits files on disk: each modification monkeypatches the imported
class/constant inside a short-lived subprocess (see ``repro/run_one.py``), so the
upstream tree stays byte-clean and ``results.json``'s ``git_dirty`` provenance
flag keeps meaning what it says.
"""

from repro.mods import budget, fusion  # noqa: F401
