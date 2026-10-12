"""``shinro.scenarios`` — the end-to-end scenario → verified ``.so`` pipeline.

The two-stage compile pipeline plus the shipped graph catalog:

- :mod:`shinro.scenarios.gen` — the zig-free stage: read a scenario TOML,
  trace + compose + lower an isolated graph pair (``graph_data.zig`` + manifest).
- :mod:`shinro.scenarios.build` — the zig stage: compile a generated graph into
  a verified, stamped ``lib*.so``.
- :mod:`shinro.scenarios.recipes` — the shipped graph recipes (KF+LQR, KF+MPC,
  policy graphs) and the live-component helpers.
- :mod:`shinro.scenarios.bake` — the statically-allocated OSQP codegen solver
  bake the MPC graphs compile against.

The generic graph builder these wrap lives in :mod:`shinro.codegen.build`; the
gates it runs live in :mod:`shinro.verification`.
"""

__all__: list[str] = []
