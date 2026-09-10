"""Deprecated — the scenario runners moved into the package.

The public simulation API now lives in :mod:`shinro.simulation.runner`
(``Scenario.run()`` / ``iter_run()``). This module re-exports the same names
so the integration suite keeps working unchanged.
"""

from shinro.simulation.runner import (  # noqa: F401
    SimResult,
    StepRecord,
    run_phase_schedule,
    run_scenario,
)
