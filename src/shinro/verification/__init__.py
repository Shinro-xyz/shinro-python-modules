"""``shinro.verification`` — correctness and drift gates for compiled artifacts.

The checks that sit between a lowered artifact and a deployment:

- :mod:`shinro.verification.oracle` — the ``.so`` vs ``interpret()`` differential
  oracle (``shinro_step`` driven through ctypes on random inputs).
- :mod:`shinro.verification.gate_a` — interpret-vs-live-numpy gate (tracer
  soundness: the traced graph must reproduce the live component bit-exactly).
- :mod:`shinro.verification.stamp` — write the deployment record (master hash
  over config ‖ graph ‖ solver ‖ binary).
- :mod:`shinro.verification.verify` — re-hash the deployed artifacts against a
  record (producer/verifier separation; the drift gate).

The size metric (:mod:`shinro.codegen.measure`) intentionally stays with the
codegen core it instruments.
"""

__all__: list[str] = []
