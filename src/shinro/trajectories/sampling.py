"""Shared schedule sampling for trajectory generators.

A generator exposes ``position_at(t) -> (pos, vel, acc)``. These helpers turn
that into the stacked ``(steps, N)`` schedules the config-driven
``from_config`` methods emit, optionally including the reference derivatives
(the ``derivatives`` opt-in). Factoring them here also removes the duplicated
sampling loop that used to live in every ``from_config``.
"""

from typing import Any


def total_duration(traj) -> float:
    """Total horizon of a generator.

    Tolerates the ``self.T`` / ``self.duration`` attribute split across the
    generators (``CubicPolynomial`` uses the latter).

    Raises:
        ValueError: If the generator exposes neither attribute.
    """
    for attr in ("T", "duration"):
        value = getattr(traj, attr, None)
        if value is not None:
            return value
    raise ValueError(f"{type(traj).__name__}: no total duration (T/duration) available")


def _keys(order: int) -> list[str]:
    keys = ["position"]
    if order >= 1:
        keys.append("velocity")
    if order >= 2:
        keys.append("acceleration")
    return keys


def _stack(bk, samples: dict[str, list]) -> dict[str, Any]:
    return {key: bk.stack(values) for key, values in samples.items()}


def sample_schedule(traj, dt: float, duration: float | None = None, order: int = 2) -> dict[str, Any]:
    """Sample one generator over ``[0, duration)`` into stacked ``(steps, N)`` arrays.

    Args:
        traj: A generator with ``position_at(t) -> (pos, vel, acc)`` and a
            backend (``traj.bk``).
        dt: Sampling period (s).
        duration: Horizon (s). Defaults to the generator's own total duration.
        order: 0 = position, 1 = +velocity, 2 = +acceleration.

    Returns:
        Dict mapping ``"position"`` (always), ``"velocity"`` (``order >= 1``),
        and ``"acceleration"`` (``order >= 2``) to ``(steps, N)`` arrays.
    """
    bk = traj.bk
    total = total_duration(traj) if duration is None else duration
    n_steps = round(total / dt)
    keys = _keys(order)
    samples: dict[str, list] = {key: [] for key in keys}
    for step in range(n_steps):
        pos, vel, acc = traj.position_at(step * dt)
        values = {"position": pos, "velocity": vel, "acceleration": acc}
        for key in keys:
            samples[key].append(values[key])
    return _stack(bk, samples)


def sample_segments(segments, dt: float, order: int = 2) -> dict[str, Any]:
    """Concatenate per-segment samples into stacked ``(steps, N)`` arrays.

    Args:
        segments: Iterable of ``(generator, duration)`` pairs, in time order.
            Each generator is sampled at ``t = k * dt`` for
            ``k < round(duration / dt)`` — the joint is emitted once, as the
            next segment's ``t = 0`` (the existing segment-schedule convention).
        dt: Sampling period (s).
        order: 0 = position, 1 = +velocity, 2 = +acceleration.

    Returns:
        Dict as in :func:`sample_schedule`.

    Raises:
        ValueError: If ``segments`` is empty.
    """
    segments = list(segments)
    if not segments:
        raise ValueError("sample_segments: no segments")
    bk = segments[0][0].bk
    keys = _keys(order)
    samples: dict[str, list] = {key: [] for key in keys}
    for traj, duration in segments:
        n_steps = round(duration / dt)
        for step in range(n_steps):
            pos, vel, acc = traj.position_at(step * dt)
            values = {"position": pos, "velocity": vel, "acceleration": acc}
            for key in keys:
                samples[key].append(values[key])
    return _stack(bk, samples)
