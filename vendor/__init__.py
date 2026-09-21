"""Vendored third-party simulator assets used only in a repo checkout.

Nothing here is shipped in the wheel or sdist (see ``MANIFEST.in``). It exists
so the LeKiwi MuJoCo model and meshes are importable by name from the repo root
(``import vendor.lekiwi_sim``), the same way ``demos`` is.

- ``lekiwi_sim.py`` — LeKiwi convenience shim (``HERE``, ``MJCF_PATH``,
  ``LeKiwiSim``); ``HERE`` resolves to this directory, so ``MJCF_PATH`` and the
  ``lekiwi-sim/meshes`` assets resolve relative to it automatically.
- ``lekiwi-sim/`` — the LeKiwi MJCF model plus mesh assets (source of the
  ``[engine].model`` referenced by ``samples/robot_config.toml``).
"""
