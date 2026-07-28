import numpy as np


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestCubicPolynomial:
    """Verify cubic polynomial: position/velocity continuity, time clamping, N-dimensional support."""

    def test_position_continuity(self, bk):
        """Position at t=0 matches p0 and at t=T matches pf."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0, 0.0])
        pf = bk.array([1.0, 2.0])
        v0 = bk.array([0.0, 0.0])
        vf = bk.array([0.0, 0.0])
        T = 2.0
        traj.generate(p0, pf, T, v0, vf)
        pos0, _, _ = traj.position_at(0.0)
        posT, _, _ = traj.position_at(T)
        assert np.allclose(_to_np(pos0, bk), _to_np(p0, bk))
        assert np.allclose(_to_np(posT, bk), _to_np(pf, bk))

    def test_velocity_continuity(self, bk):
        """Velocity at t=0 matches v0 and at t=T matches vf."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.5])
        vf = bk.array([0.0])
        T = 2.0
        traj.generate(p0, pf, T, v0, vf)
        _, vel0, _ = traj.position_at(0.0)
        _, velT, _ = traj.position_at(T)
        assert np.allclose(_to_np(vel0, bk), _to_np(v0, bk))
        assert np.allclose(_to_np(velT, bk), _to_np(vf, bk))

    def test_acceleration_not_constrained(self, bk):
        """Cubic polynomial does NOT constrain acceleration at the boundaries."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0])
        vf = bk.array([0.0])
        T = 2.0
        traj.generate(p0, pf, T, v0, vf)
        _, _, acc0 = traj.position_at(0.0)
        _, _, accT = traj.position_at(T)
        assert not np.allclose(_to_np(acc0, bk), 0.0)
        assert not np.allclose(_to_np(accT, bk), 0.0)

    def test_time_clamped(self, bk):
        """Time outside [0, T] is clamped to the nearest endpoint."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0])
        vf = bk.array([0.0])
        T = 2.0
        traj.generate(p0, pf, T, v0, vf)
        pos_before, _, _ = traj.position_at(-1.0)
        pos_after, _, _ = traj.position_at(3.0)
        assert np.allclose(_to_np(pos_before, bk), _to_np(p0, bk))
        assert np.allclose(_to_np(pos_after, bk), _to_np(pf, bk))

    def test_ndimensional(self, bk):
        """Cubic polynomial supports N-dimensional positions."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0, 1.0, 2.0])
        pf = bk.array([3.0, 4.0, 5.0])
        v0 = bk.array([0.0, 0.0, 0.0])
        vf = bk.array([0.0, 0.0, 0.0])
        T = 1.0
        traj.generate(p0, pf, T, v0, vf)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (3,)


class TestQuinticPolynomial:
    """Verify quintic polynomial: position/velocity/acceleration continuity, minimum-jerk, time clamping."""

    def test_position_continuity(self, bk):
        """Position at t=0 matches p0 and at t=T matches pf."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0, 0.0])
        pf = bk.array([1.0, 2.0])
        T = 2.0
        traj.generate(p0, pf, T)
        pos0, _, _ = traj.position_at(0.0)
        posT, _, _ = traj.position_at(T)
        assert np.allclose(_to_np(pos0, bk), _to_np(p0, bk))
        assert np.allclose(_to_np(posT, bk), _to_np(pf, bk))

    def test_velocity_continuity(self, bk):
        """Velocity at t=0 matches v0 and at t=T matches vf."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.5])
        vf = bk.array([0.0])
        T = 2.0
        traj.generate(p0, pf, T, start_vel=v0, end_vel=vf)
        _, vel0, _ = traj.position_at(0.0)
        _, velT, _ = traj.position_at(T)
        assert np.allclose(_to_np(vel0, bk), _to_np(v0, bk))
        assert np.allclose(_to_np(velT, bk), _to_np(vf, bk))

    def test_acceleration_continuity(self, bk):
        """Acceleration at t=0 matches a0 and at t=T matches af."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        a0 = bk.array([0.2])
        af = bk.array([-0.1])
        T = 2.0
        traj.generate(p0, pf, T, start_acc=a0, end_acc=af)
        _, _, acc0 = traj.position_at(0.0)
        _, _, accT = traj.position_at(T)
        assert np.allclose(_to_np(acc0, bk), _to_np(a0, bk))
        assert np.allclose(_to_np(accT, bk), _to_np(af, bk))

    def test_minimum_jerk_rest_to_rest(self, bk):
        """Rest-to-rest quintic matches the minimum-jerk formula p(s) = p0 + (pf-p0)(10s^3 - 15s^4 + 6s^5)."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        T = 1.0
        traj.generate(p0, pf, T)
        s = 0.5
        t = s * T
        pos, _, _ = traj.position_at(t)
        pos_val = _to_np(pos, bk)[0]
        pos_expected = 0.0 + (1.0 - 0.0) * (10 * s**3 - 15 * s**4 + 6 * s**5)
        assert np.allclose(pos_val, pos_expected)

    def test_time_clamped(self, bk):
        """Time outside [0, T] is clamped to the nearest endpoint."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        T = 2.0
        traj.generate(p0, pf, T)
        pos_before, _, _ = traj.position_at(-1.0)
        pos_after, _, _ = traj.position_at(3.0)
        assert np.allclose(_to_np(pos_before, bk), _to_np(p0, bk))
        assert np.allclose(_to_np(pos_after, bk), _to_np(pf, bk))

    def test_ndimensional(self, bk):
        """Quintic polynomial supports N-dimensional positions."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0, 1.0, 2.0])
        pf = bk.array([3.0, 4.0, 5.0])
        T = 1.0
        traj.generate(p0, pf, T)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (3,)
