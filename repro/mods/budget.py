"""Sensitivity arm — move the H1 false-alarm budget.

This is **not** the paper's modification (that is ``repro.mods.fusion``). It is a
sensitivity check on the reproduced number: the headline table is stated at a 5%
healthy-validation false-alarm budget, and the paper itself warns that the
*realised* rate can miss the budget. Re-running the unchanged pipeline at a
tighter and a looser budget answers "how much of the headline row is an artefact
of the operating point?" with the same code, same seed, same data.
"""

from __future__ import annotations


def apply(budget: float) -> None:
    from derail.experiments import run_experiment

    budget = float(budget)
    if not 0.0 < budget < 1.0:
        raise ValueError(f"false-alarm budget must be in (0,1), got {budget}")
    run_experiment.FA_BUDGET = budget
