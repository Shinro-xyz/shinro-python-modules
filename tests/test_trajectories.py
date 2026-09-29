import numpy as np
import pytest


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestCubicPolynomial:
    """Verify cubic polynomial: position/velocity continuity, time clamping, N-dimensional support."""

    def test_position_continuity(self, bk):
        """Position at t=0 matches p0 and at t=T matches pf."""
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
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

    def test_cubic_coefficient_formula(self, bk):
        """Cubic coefficients match the closed-form solution a0=p0, a1=v0, a2=3Δp/T² - (2v0+vf)/T, a3=-2Δp/T³ + (v0+vf)/T²."""
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([1.0])
        pf = bk.array([4.0])
        v0 = bk.array([0.5])
        vf = bk.array([-0.3])
        T = 2.0
        traj.generate(p0, pf, T, v0, vf)
        dp = _to_np(pf - p0, bk)[0]
        a0_expected = 1.0
        a1_expected = 0.5
        a2_expected = (3 * dp - T * (2 * 0.5 + (-0.3))) / (T ** 2)
        a3_expected = (-2 * dp + T * (0.5 + (-0.3))) / (T ** 3)
        assert np.allclose(_to_np(traj.a0, bk)[0], a0_expected)
        assert np.allclose(_to_np(traj.a1, bk)[0], a1_expected)
        assert np.allclose(_to_np(traj.a2, bk)[0], a2_expected)
        assert np.allclose(_to_np(traj.a3, bk)[0], a3_expected)

    def test_cubic_velocity_at_endpoints(self, bk):
        """Velocity at t=0 and t=T matches the specified boundary conditions."""
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.5])
        vf = bk.array([-0.3])
        T = 2.0
        traj.generate(p0, pf, T, v0, vf)
        _, vel0, _ = traj.position_at(0.0)
        _, velT, _ = traj.position_at(T)
        assert np.allclose(_to_np(vel0, bk)[0], 0.5)
        assert np.allclose(_to_np(velT, bk)[0], -0.3)

    def test_cubic_derivative_of_position_is_velocity(self, bk):
        """The velocity returned is the analytic derivative of the position polynomial."""
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([2.0])
        v0 = bk.array([0.5])
        vf = bk.array([0.0])
        T = 2.0
        traj.generate(p0, pf, T, v0, vf)
        eps = 1e-6
        pos_plus, _, _ = traj.position_at(1.0 + eps)
        pos_minus, _, _ = traj.position_at(1.0 - eps)
        _, vel, _ = traj.position_at(1.0)
        numerical_deriv = (_to_np(pos_plus, bk)[0] - _to_np(pos_minus, bk)[0]) / (2 * eps)
        assert np.allclose(_to_np(vel, bk)[0], numerical_deriv, atol=1e-4)

    def test_velocity_continuity(self, bk):
        """Velocity at t=0 matches v0 and at t=T matches vf."""
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
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
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
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
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
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
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
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
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0, 0.0])
        pf = bk.array([1.0, 2.0])
        T = 2.0
        traj.generate(p0, pf, T)
        pos0, _, _ = traj.position_at(0.0)
        posT, _, _ = traj.position_at(T)
        assert np.allclose(_to_np(pos0, bk), _to_np(p0, bk))
        assert np.allclose(_to_np(posT, bk), _to_np(pf, bk))

    def test_quintic_solves_6x6_system(self, bk):
        """Quintic coefficients satisfy the 6 boundary conditions exactly."""
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([1.0])
        pf = bk.array([4.0])
        v0 = bk.array([0.5])
        vf = bk.array([-0.3])
        a0 = bk.array([0.2])
        af = bk.array([-0.1])
        T = 2.0
        traj.generate(p0, pf, T, v0, vf, a0, af)
        pos0, vel0, acc0 = traj.position_at(0.0)
        posT, velT, accT = traj.position_at(T)
        assert np.allclose(_to_np(pos0, bk)[0], 1.0)
        assert np.allclose(_to_np(posT, bk)[0], 4.0)
        assert np.allclose(_to_np(vel0, bk)[0], 0.5)
        assert np.allclose(_to_np(velT, bk)[0], -0.3)
        assert np.allclose(_to_np(acc0, bk)[0], 0.2)
        assert np.allclose(_to_np(accT, bk)[0], -0.1)

    def test_quintic_minimum_jerk_rest_to_rest(self, bk):
        """Rest-to-rest quintic matches the minimum-jerk formula p(s) = p0 + (pf-p0)(10s^3 - 15s^4 + 6s^5)."""
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
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

    def test_quintic_derivative_of_position_is_velocity(self, bk):
        """The velocity returned is the analytic derivative of the position polynomial."""
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([2.0])
        T = 2.0
        traj.generate(p0, pf, T)
        eps = 1e-6
        pos_plus, _, _ = traj.position_at(1.0 + eps)
        pos_minus, _, _ = traj.position_at(1.0 - eps)
        _, vel, _ = traj.position_at(1.0)
        numerical_deriv = (_to_np(pos_plus, bk)[0] - _to_np(pos_minus, bk)[0]) / (2 * eps)
        assert np.allclose(_to_np(vel, bk)[0], numerical_deriv, atol=1e-4)

    def test_quintic_derivative_of_velocity_is_acceleration(self, bk):
        """The acceleration returned is the analytic derivative of the velocity polynomial."""
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([2.0])
        T = 2.0
        traj.generate(p0, pf, T)
        eps = 1e-6
        _, vel_plus, _ = traj.position_at(1.0 + eps)
        _, vel_minus, _ = traj.position_at(1.0 - eps)
        _, _, acc = traj.position_at(1.0)
        numerical_deriv = (_to_np(vel_plus, bk)[0] - _to_np(vel_minus, bk)[0]) / (2 * eps)
        assert np.allclose(_to_np(acc, bk)[0], numerical_deriv, atol=1e-4)

    def test_velocity_continuity(self, bk):
        """Velocity at t=0 matches v0 and at t=T matches vf."""
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
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
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
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
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
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
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
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
        from shinro.trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0, 1.0, 2.0])
        pf = bk.array([3.0, 4.0, 5.0])
        T = 1.0
        traj.generate(p0, pf, T)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (3,)


class TestLissajous:
    """Verify Lissajous: endpoints, zero boundary velocity, R orientation, derivatives, config."""

    @staticmethod
    def _rot_z(theta):
        """Rotation matrix about z (world-from-local)."""
        c, s = np.cos(theta), np.sin(theta)
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]

    def _make(self, bk, start, end, T, k, R=None):
        from shinro.trajectories.lissajous import Lissajous
        traj = Lissajous(k, R=R, backend=bk)
        traj.generate(bk.array(start), bk.array(end), T)
        return traj

    def test_position_endpoints(self, bk):
        """Position at t=0 matches start and at t=T matches end."""
        traj = self._make(bk, [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 2.0, [1.0, 1.0, 0.0])
        pos0, _, _ = traj.position_at(0.0)
        posT, _, _ = traj.position_at(2.0)
        assert np.allclose(_to_np(pos0, bk), [1.0, 0.0, 0.0])
        assert np.allclose(_to_np(posT, bk), [-1.0, 0.0, 0.0])

    def test_zero_boundary_velocity(self, bk):
        """Odd harmonics make the velocity zero at both ends."""
        traj = self._make(bk, [0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], 2.0, [1.0, 2.0, 0.0])
        _, vel0, _ = traj.position_at(0.0)
        _, velT, _ = traj.position_at(2.0)
        assert np.allclose(_to_np(vel0, bk), 0.0, atol=1e-12)
        assert np.allclose(_to_np(velT, bk), 0.0, atol=1e-12)

    def test_frequencies(self, bk):
        """Per-axis frequencies are (2k_i + 1)*pi/T."""
        k = [1.0, 2.0, 0.0]
        T = 2.0
        traj = self._make(bk, [0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], T, k)
        expected = (2 * np.array(k) + 1) * np.pi / T
        assert np.allclose(_to_np(traj.omegas, bk), expected)

    def test_rotated_R_hits_endpoints(self, bk):
        """A non-identity rotation still interpolates start and end exactly."""
        k = [1.0, 2.0, 0.0]
        traj = self._make(
            bk, [0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], 2.0, k, R=self._rot_z(np.pi / 4)
        )
        pos0, _, _ = traj.position_at(0.0)
        posT, _, _ = traj.position_at(2.0)
        assert np.allclose(_to_np(pos0, bk), [0.5, 0.5, 0.0])
        assert np.allclose(_to_np(posT, bk), [-0.5, -0.5, 0.0])

    def test_R_orients_the_figure(self, bk):
        """R changes the interior path (mixing per-axis frequencies) while endpoints hold."""
        k = [1.0, 2.0, 0.0]
        start, end = [0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]
        identity = self._make(bk, start, end, 2.0, k)
        rotated = self._make(bk, start, end, 2.0, k, R=self._rot_z(np.pi / 4))
        assert not np.allclose(
            _to_np(identity.position_at(0.3)[0], bk),
            _to_np(rotated.position_at(0.3)[0], bk),
        )

    def test_derivative_of_position_is_velocity(self, bk):
        """The returned velocity is the analytic derivative of the position."""
        traj = self._make(
            bk, [0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], 2.0, [1.0, 2.0, 0.0], R=self._rot_z(np.pi / 4)
        )
        eps = 1e-6
        pos_plus = _to_np(traj.position_at(0.7 + eps)[0], bk)
        pos_minus = _to_np(traj.position_at(0.7 - eps)[0], bk)
        vel = _to_np(traj.position_at(0.7)[1], bk)
        assert np.allclose((pos_plus - pos_minus) / (2 * eps), vel, atol=1e-4)

    def test_derivative_of_velocity_is_acceleration(self, bk):
        """The returned acceleration is the analytic derivative of the velocity."""
        traj = self._make(bk, [0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], 2.0, [1.0, 2.0, 0.0])
        eps = 1e-6
        vel_plus = _to_np(traj.position_at(0.7 + eps)[1], bk)
        vel_minus = _to_np(traj.position_at(0.7 - eps)[1], bk)
        acc = _to_np(traj.position_at(0.7)[2], bk)
        assert np.allclose((vel_plus - vel_minus) / (2 * eps), acc, atol=1e-4)

    def test_time_clamped(self, bk):
        """Time outside [0, T] is clamped to the nearest endpoint."""
        traj = self._make(bk, [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 2.0, [1.0, 1.0, 0.0])
        pos_before, _, _ = traj.position_at(-1.0)
        pos_after, _, _ = traj.position_at(3.0)
        assert np.allclose(_to_np(pos_before, bk), [1.0, 0.0, 0.0])
        assert np.allclose(_to_np(pos_after, bk), [-1.0, 0.0, 0.0])

    def test_ndimensional(self, bk):
        """Lissajous supports N-dimensional positions with a matching identity R."""
        traj = self._make(bk, [1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], 1.0, [0.0, 0.0, 0.0, 0.0])
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (4,)

    def test_k_length_mismatch_raises(self, bk):
        """A k vector that disagrees with the position dimension is a loud error."""
        from shinro.trajectories.lissajous import Lissajous
        traj = Lissajous([1.0, 1.0], backend=bk)
        with pytest.raises(ValueError, match="k has 2 entries"):
            traj.generate(bk.array([0.0, 0.0, 0.0]), bk.array([1.0, 1.0, 1.0]), 1.0)

    def test_non_rotation_R_raises(self, bk):
        """A non-orthogonal R breaks the endpoint guarantee and is rejected."""
        from shinro.trajectories.lissajous import Lissajous
        traj = Lissajous([1.0, 1.0, 0.0], R=[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], backend=bk)
        with pytest.raises(ValueError, match="rotation matrix"):
            traj.generate(bk.array([1.0, 0.0, 0.0]), bk.array([-1.0, 0.0, 0.0]), 1.0)

    def test_from_config_schedule(self, bk):
        """from_config samples the figure into a (steps, N) waypoint schedule."""
        from shinro.trajectories.lissajous import Lissajous
        schedule = Lissajous.from_config(
            {
                "type": "lissajous",
                "dt": 0.1,
                "duration": 1.0,
                "start": [0.5, 0.5, 0.0],
                "end": [-0.5, -0.5, 0.0],
                "k": [1.0, 2.0, 0.0],
            },
            backend=bk,
        )
        arr = _to_np(schedule, bk)
        assert arr.shape == (10, 3)
        assert np.allclose(arr[0], [0.5, 0.5, 0.0])


class TestBezierCurve:
    """Verify BezierCurve: endpoints, boundary velocities, derivatives, validation."""

    CUBIC = ((0.0, 0.0, 0.0), (0.4, 0.6, 0.0), (0.8, -0.6, 0.0), (1.2, 0.0, 0.0))

    def _make(self, bk, points, T):
        from shinro.trajectories.bezier_curve import BezierCurve
        traj = BezierCurve(points, backend=bk)
        traj.generate(duration=T)
        return traj

    def test_endpoints_interpolated(self, bk):
        """B(0) is the first control point and B(T) is the last."""
        traj = self._make(bk, self.CUBIC, 2.0)
        pos0, _, _ = traj.position_at(0.0)
        posT, _, _ = traj.position_at(2.0)
        assert np.allclose(_to_np(pos0, bk), self.CUBIC[0])
        assert np.allclose(_to_np(posT, bk), self.CUBIC[-1])

    def test_boundary_velocities(self, bk):
        """B'(0) = n/T (P1 - P0), and B'(T) uses the last two control points."""
        traj = self._make(bk, self.CUBIC, 2.0)
        n = len(self.CUBIC) - 1
        _, vel0, _ = traj.position_at(0.0)
        _, velT, _ = traj.position_at(2.0)
        expected0 = (n / 2.0) * (np.array(self.CUBIC[1]) - np.array(self.CUBIC[0]))
        expectedT = (n / 2.0) * (np.array(self.CUBIC[-1]) - np.array(self.CUBIC[-2]))
        assert np.allclose(_to_np(vel0, bk), expected0)
        assert np.allclose(_to_np(velT, bk), expectedT)

    def test_known_line(self, bk):
        """Collinear control points [[0],[1],[2]] give B(s) = 2s exactly."""
        traj = self._make(bk, [[0.0], [1.0], [2.0]], 4.0)
        pos, vel, acc = traj.position_at(1.0)
        assert np.allclose(_to_np(pos, bk)[0], 0.5)
        assert np.allclose(_to_np(vel, bk)[0], 0.5)
        assert np.allclose(_to_np(acc, bk)[0], 0.0)

    def test_duplicate_endpoints_zero_boundary_velocity(self, bk):
        """Duplicating the end points pins the boundary velocity to zero."""
        traj = self._make(bk, [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], 2.0)
        _, vel0, _ = traj.position_at(0.0)
        _, velT, _ = traj.position_at(2.0)
        assert np.allclose(_to_np(vel0, bk), 0.0)
        assert np.allclose(_to_np(velT, bk), 0.0)

    def test_derivative_of_position_is_velocity(self, bk):
        """The returned velocity is the analytic derivative of the position."""
        traj = self._make(bk, self.CUBIC, 2.0)
        eps = 1e-6
        pos_plus = _to_np(traj.position_at(0.7 + eps)[0], bk)
        pos_minus = _to_np(traj.position_at(0.7 - eps)[0], bk)
        vel = _to_np(traj.position_at(0.7)[1], bk)
        assert np.allclose((pos_plus - pos_minus) / (2 * eps), vel, atol=1e-4)

    def test_derivative_of_velocity_is_acceleration(self, bk):
        """The returned acceleration is the analytic derivative of the velocity."""
        traj = self._make(bk, self.CUBIC, 2.0)
        eps = 1e-6
        vel_plus = _to_np(traj.position_at(0.7 + eps)[1], bk)
        vel_minus = _to_np(traj.position_at(0.7 - eps)[1], bk)
        acc = _to_np(traj.position_at(0.7)[2], bk)
        assert np.allclose((vel_plus - vel_minus) / (2 * eps), acc, atol=1e-4)

    def test_degree1_zero_acceleration(self, bk):
        """A two-point (straight) curve has constant velocity and zero acceleration."""
        traj = self._make(bk, [[0.0, 0.0], [2.0, 0.0]], 2.0)
        pos, vel, acc = traj.position_at(1.0)
        assert np.allclose(_to_np(vel, bk), [1.0, 0.0])
        assert np.allclose(_to_np(acc, bk), 0.0)
        assert _to_np(pos, bk).shape == (2,)

    def test_time_clamped(self, bk):
        """Time outside [0, T] is clamped to the nearest endpoint."""
        traj = self._make(bk, self.CUBIC, 2.0)
        pos_before, _, _ = traj.position_at(-1.0)
        pos_after, _, _ = traj.position_at(5.0)
        assert np.allclose(_to_np(pos_before, bk), self.CUBIC[0])
        assert np.allclose(_to_np(pos_after, bk), self.CUBIC[-1])

    def test_ndimensional(self, bk):
        """The spatial dimension is inferred from the control points."""
        traj = self._make(bk, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], 1.0)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (3,)

    def test_ragged_control_points_raise(self, bk):
        """Control points of differing dimension are a loud error."""
        from shinro.trajectories.bezier_curve import BezierCurve
        traj = BezierCurve([[0.0, 0.0], [1.0, 1.0, 1.0]], backend=bk)
        with pytest.raises(ValueError, match="same dimension"):
            traj.generate(duration=1.0)

    def test_too_few_points_raise(self, bk):
        """A single control point is not a curve."""
        from shinro.trajectories.bezier_curve import BezierCurve
        traj = BezierCurve([[0.0, 0.0]], backend=bk)
        with pytest.raises(ValueError, match="at least 2"):
            traj.generate(duration=1.0)

    def test_bad_duration_raises(self, bk):
        """A non-positive (or missing) duration is a loud error."""
        from shinro.trajectories.bezier_curve import BezierCurve
        traj = BezierCurve(self.CUBIC, backend=bk)
        with pytest.raises(ValueError, match="duration must be positive"):
            traj.generate(duration=0.0)
        with pytest.raises(ValueError, match="duration must be positive"):
            traj.generate()

    def test_start_mismatch_raises(self, bk):
        """A declared start that disagrees with the first control point raises."""
        from shinro.trajectories.bezier_curve import BezierCurve
        traj = BezierCurve(self.CUBIC, backend=bk)
        with pytest.raises(ValueError, match="disagrees"):
            traj.generate(start_position=[9.0, 9.0, 9.0], duration=1.0)

    def test_from_config_schedule(self, bk):
        """from_config samples the curve into a (steps, d) waypoint schedule."""
        from shinro.trajectories.bezier_curve import BezierCurve
        schedule = BezierCurve.from_config(
            {
                "type": "bezier",
                "dt": 0.1,
                "duration": 1.0,
                "control_points": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            },
            backend=bk,
        )
        arr = _to_np(schedule, bk)
        assert arr.shape == (10, 2)
        assert np.allclose(arr[0], [0.0, 0.0])


class TestBSpline:
    """Verify BSpline: clamped cubic == Bezier, Cox-de Boor derivatives, validation."""

    # Clamped cubic knot vector over 4 control points. The curve domain is
    # [u_p, u_n+1] = [0, 1], which is also the duration used below.
    KNOTS = (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    CUBIC = ((0.0, 0.0, 0.0), (0.4, 0.6, 0.0), (0.8, -0.6, 0.0), (1.2, 0.0, 0.0))

    def _make(self, bk, degree, points, knots, T):
        from shinro.trajectories.b_spline import BSpline
        traj = BSpline(degree, points, knots, backend=bk)
        traj.generate(duration=T)
        return traj

    def test_clamped_cubic_matches_bezier(self, bk):
        """A clamped cubic B-spline over 4 control points IS a cubic Bezier."""
        from shinro.trajectories.bezier_curve import BezierCurve
        spline = self._make(bk, 3, self.CUBIC, self.KNOTS, 1.0)
        bez = BezierCurve(self.CUBIC, backend=bk)
        bez.generate(duration=1.0)
        for t in (0.0, 0.2, 0.5, 0.8, 1.0):
            sp, sv, sa = spline.position_at(t)
            bp, bv, ba = bez.position_at(t)
            assert np.allclose(_to_np(sp, bk), _to_np(bp, bk))
            assert np.allclose(_to_np(sv, bk), _to_np(bv, bk))
            assert np.allclose(_to_np(sa, bk), _to_np(ba, bk))

    def test_endpoints_interpolated(self, bk):
        """A clamped knot vector interpolates C(0) = P0 and C(T) = P_last."""
        traj = self._make(bk, 3, self.CUBIC, self.KNOTS, 1.0)
        pos0, _, _ = traj.position_at(0.0)
        posT, _, _ = traj.position_at(1.0)
        assert np.allclose(_to_np(pos0, bk), self.CUBIC[0])
        assert np.allclose(_to_np(posT, bk), self.CUBIC[-1])

    def test_boundary_velocities(self, bk):
        """C'(0) = p/T (P1 - P0); C'(T) uses the last two control points."""
        traj = self._make(bk, 3, self.CUBIC, self.KNOTS, 1.0)
        _, vel0, _ = traj.position_at(0.0)
        _, velT, _ = traj.position_at(1.0)
        expected0 = 3.0 * (np.array(self.CUBIC[1]) - np.array(self.CUBIC[0]))
        expectedT = 3.0 * (np.array(self.CUBIC[-1]) - np.array(self.CUBIC[-2]))
        assert np.allclose(_to_np(vel0, bk), expected0)
        assert np.allclose(_to_np(velT, bk), expectedT)

    def test_derivative_of_position_is_velocity(self, bk):
        """The returned velocity is the analytic derivative of the position."""
        traj = self._make(bk, 3, self.CUBIC, self.KNOTS, 1.0)
        eps = 1e-6
        pos_plus = _to_np(traj.position_at(0.6 + eps)[0], bk)
        pos_minus = _to_np(traj.position_at(0.6 - eps)[0], bk)
        vel = _to_np(traj.position_at(0.6)[1], bk)
        assert np.allclose((pos_plus - pos_minus) / (2 * eps), vel, atol=1e-4)

    def test_derivative_of_velocity_is_acceleration(self, bk):
        """The returned acceleration is the analytic derivative of the velocity."""
        traj = self._make(bk, 3, self.CUBIC, self.KNOTS, 1.0)
        eps = 1e-6
        vel_plus = _to_np(traj.position_at(0.6 + eps)[1], bk)
        vel_minus = _to_np(traj.position_at(0.6 - eps)[1], bk)
        acc = _to_np(traj.position_at(0.6)[2], bk)
        assert np.allclose((vel_plus - vel_minus) / (2 * eps), acc, atol=1e-4)

    def test_partition_of_unity(self, bk):
        """The Cox-de Boor basis functions sum to one over the domain."""
        traj = self._make(bk, 3, self.CUBIC, self.KNOTS, 1.0)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            N = _to_np(traj._all_basis(t, 3), bk)
            assert np.isclose(np.sum(N), 1.0)

    def test_degree1_line(self, bk):
        """A degree-1 spline over collinear points is a straight line, zero acceleration."""
        traj = self._make(bk, 1, [[0.0], [1.0], [2.0]], [0.0, 0.0, 1.0, 2.0, 2.0], 2.0)
        pos, vel, acc = traj.position_at(1.0)
        assert np.allclose(_to_np(pos, bk)[0], 1.0)
        assert np.allclose(_to_np(vel, bk)[0], 1.0)
        assert np.allclose(_to_np(acc, bk)[0], 0.0)

    def test_degree0_piecewise_constant(self, bk):
        """A degree-0 spline is a step function with zero velocity and acceleration."""
        traj = self._make(bk, 0, [[1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0], 3.0)
        pos, vel, acc = traj.position_at(1.5)
        assert np.allclose(_to_np(pos, bk)[0], 2.0)
        assert np.allclose(_to_np(vel, bk), 0.0)
        assert np.allclose(_to_np(acc, bk), 0.0)

    def test_duplicate_endpoints_zero_boundary_velocity(self, bk):
        """Repeating the end control points pins the boundary velocity to zero."""
        pts = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        traj = self._make(bk, 3, pts, self.KNOTS, 1.0)
        _, vel0, _ = traj.position_at(0.0)
        _, velT, _ = traj.position_at(1.0)
        assert np.allclose(_to_np(vel0, bk), 0.0)
        assert np.allclose(_to_np(velT, bk), 0.0)

    def test_uniform_knots_derivative_consistency(self, bk):
        """On a non-clamped vector the reduced-knot derivatives match finite differences."""
        knots = list(range(11))
        pts = [[0.0], [1.0], [-1.0], [2.0], [0.0], [3.0], [1.0]]
        traj = self._make(bk, 3, pts, knots, 10.0)
        t, eps = 5.0, 1e-6
        pos_plus = _to_np(traj.position_at(t + eps)[0], bk)
        pos_minus = _to_np(traj.position_at(t - eps)[0], bk)
        vel = _to_np(traj.position_at(t)[1], bk)
        assert np.allclose((pos_plus - pos_minus) / (2 * eps), vel, atol=1e-4)

    def test_time_clamped(self, bk):
        """Time outside [0, T] is clamped to the nearest endpoint."""
        traj = self._make(bk, 3, self.CUBIC, self.KNOTS, 1.0)
        pos_before, _, _ = traj.position_at(-1.0)
        pos_after, _, _ = traj.position_at(9.0)
        assert np.allclose(_to_np(pos_before, bk), self.CUBIC[0])
        assert np.allclose(_to_np(pos_after, bk), self.CUBIC[-1])

    def test_ndimensional(self, bk):
        """The spatial dimension is inferred from the control points."""
        traj = self._make(bk, 3, self.CUBIC, self.KNOTS, 1.0)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (3,)

    def test_ragged_control_points_raise(self, bk):
        """Control points of differing dimension are a loud error."""
        from shinro.trajectories.b_spline import BSpline
        traj = BSpline(3, [[0.0, 0.0], [1.0, 1.0, 1.0]], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], backend=bk)
        with pytest.raises(ValueError, match="same dimension"):
            traj.generate(duration=1.0)

    def test_too_few_points_raise(self, bk):
        """A single control point is not a curve."""
        from shinro.trajectories.b_spline import BSpline
        traj = BSpline(1, [[0.0, 0.0]], [0.0, 0.0, 1.0, 1.0], backend=bk)
        with pytest.raises(ValueError, match="at least 2"):
            traj.generate(duration=1.0)

    def test_bad_duration_raises(self, bk):
        """A non-positive duration is a loud error."""
        from shinro.trajectories.b_spline import BSpline
        traj = BSpline(3, self.CUBIC, self.KNOTS, backend=bk)
        with pytest.raises(ValueError, match="duration must be positive"):
            traj.generate(duration=0.0)
        with pytest.raises(ValueError, match="duration must be positive"):
            traj.generate(duration=-1.0)

    def test_knot_count_mismatch_raises(self, bk):
        """A knot vector that is not len(control_points) + degree + 1 is an error."""
        from shinro.trajectories.b_spline import BSpline
        traj = BSpline(3, self.CUBIC, self.KNOTS[:-1], backend=bk)
        with pytest.raises(ValueError, match="knot vector must hold"):
            traj.generate(duration=1.0)

    def test_non_monotone_knots_raise(self, bk):
        """A nondecreasing knot vector is required."""
        from shinro.trajectories.b_spline import BSpline
        traj = BSpline(1, [[0.0], [1.0]], [0.0, 2.0, 1.0, 2.0], backend=bk)
        with pytest.raises(ValueError, match="nondecreasing"):
            traj.generate(duration=1.0)

    def test_negative_degree_raises(self, bk):
        """A negative polynomial degree is a loud error."""
        from shinro.trajectories.b_spline import BSpline
        traj = BSpline(-1, [[0.0], [1.0]], [0.0, 1.0], backend=bk)
        with pytest.raises(ValueError, match="degree must be non-negative"):
            traj.generate(duration=1.0)

    def test_from_config_schedule(self, bk):
        """from_config samples the curve into a (steps, d) waypoint schedule."""
        from shinro.trajectories.b_spline import BSpline
        schedule = BSpline.from_config(
            {
                "type": "bspline",
                "dt": 0.1,
                "duration": 1.0,
                "degree": 3,
                "control_points": [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            },
            backend=bk,
        )
        arr = _to_np(schedule, bk)
        assert arr.shape == (10, 2)
        assert np.allclose(arr[0], [0.0, 0.0])


class TestCatmullRom:
    """Verify CatmullRom: C1 pass-through, tangent formula, endpoints, validation."""

    START = (0.0, 0.0)
    WPS = ((1.0, 1.0), (2.0, -1.0), (3.0, 1.0))
    DURATIONS = (1.0, 1.0, 1.0)

    def _make(self, bk, start=None, wps=None, durations=None, endpoint_tangent="zero"):
        from shinro.trajectories.catmull_rom import CatmullRom
        traj = CatmullRom(backend=bk)
        traj.generate(
            [self.START if start is None else start, *(self.WPS if wps is None else wps)],
            list(self.DURATIONS if durations is None else durations),
            endpoint_tangent,
        )
        return traj

    @staticmethod
    def _points(start, wps):
        return [start] + list(wps)

    def test_passes_through_waypoints(self, bk):
        """The curve hits the start and every waypoint at its cumulative time."""
        traj = self._make(bk)
        pts = self._points(self.START, self.WPS)
        assert np.allclose(_to_np(traj.position_at(0.0)[0], bk), pts[0])
        t = 0.0
        for i, d in enumerate(self.DURATIONS):
            t += d
            assert np.allclose(_to_np(traj.position_at(t)[0], bk), pts[i + 1])

    def test_interior_tangent_formula(self, bk):
        """Interior velocity is (P_{i+1} - P_{i-1}) / (d_{i-1} + d_i)."""
        traj = self._make(bk)
        pts = self._points(self.START, self.WPS)
        for i in (1, 2):
            want = (np.array(pts[i + 1]) - np.array(pts[i - 1])) / (
                self.DURATIONS[i - 1] + self.DURATIONS[i]
            )
            got = _to_np(traj.position_at(sum(self.DURATIONS[:i]))[1], bk)
            assert np.allclose(got, want)

    def test_velocity_continuous_at_interior_waypoints(self, bk):
        """The left/right velocities agree at every interior waypoint (C1)."""
        traj = self._make(bk)
        segs = [seg for (_, _, seg) in traj._segments]
        for i in range(len(segs) - 1):
            v_left = _to_np(segs[i].position_at(self.DURATIONS[i])[1], bk)
            v_right = _to_np(segs[i + 1].position_at(0.0)[1], bk)
            assert np.allclose(v_left, v_right)

    def test_velocity_matches_finite_difference_across_joint(self, bk):
        """A central difference across a C1 joint matches the analytic velocity."""
        traj = self._make(bk)
        t, eps = self.DURATIONS[0], 1e-6
        p_plus = _to_np(traj.position_at(t + eps)[0], bk)
        p_minus = _to_np(traj.position_at(t - eps)[0], bk)
        vel = _to_np(traj.position_at(t)[1], bk)
        assert np.allclose((p_plus - p_minus) / (2 * eps), vel, atol=1e-3)

    def test_zero_endpoint_velocity_default(self, bk):
        """The default endpoint tangent is zero (rest-to-rest ends)."""
        traj = self._make(bk)
        assert np.allclose(_to_np(traj.position_at(0.0)[1], bk), 0.0)
        assert np.allclose(_to_np(traj.position_at(traj.T)[1], bk), 0.0)

    def test_one_sided_endpoint_velocity(self, bk):
        """``one_sided`` uses the classic non-zero Catmull-Rom end tangents."""
        traj = self._make(bk, endpoint_tangent="one_sided")
        pts = self._points(self.START, self.WPS)
        v0 = (np.array(pts[1]) - np.array(pts[0])) / self.DURATIONS[0]
        vn = (np.array(pts[-1]) - np.array(pts[-2])) / self.DURATIONS[-1]
        assert np.allclose(_to_np(traj.position_at(0.0)[1], bk), v0)
        assert np.allclose(_to_np(traj.position_at(traj.T)[1], bk), vn)

    def test_collinear_equal_spacing_constant_velocity(self, bk):
        """Collinear, equal-duration waypoints give a straight constant-velocity line."""
        traj = self._make(
            bk,
            start=[0.0],
            wps=[[1.0], [2.0], [3.0]],
            durations=[1.0, 1.0, 1.0],
            endpoint_tangent="one_sided",
        )
        for t in (0.0, 0.5, 1.5, 2.5, 3.0):
            pos, vel, acc = traj.position_at(t)
            assert np.allclose(_to_np(pos, bk)[0], t)
            assert np.allclose(_to_np(vel, bk)[0], 1.0)
            assert np.allclose(_to_np(acc, bk)[0], 0.0)

    def test_ndimensional(self, bk):
        """Position/velocity/acceleration keep the waypoint dimension."""
        traj = self._make(
            bk,
            start=[0.0, 0.0, 0.0],
            wps=[[1.0, 1.0, 1.0], [2.0, 0.0, 2.0]],
            durations=[1.0, 1.0],
        )
        pos, vel, acc = traj.position_at(1.0)
        assert _to_np(pos, bk).shape == (3,)
        assert _to_np(vel, bk).shape == (3,)
        assert _to_np(acc, bk).shape == (3,)

    def test_too_few_waypoints_raise(self, bk):
        """A single waypoint is not a curve."""
        from shinro.trajectories.catmull_rom import CatmullRom
        traj = CatmullRom(backend=bk)
        with pytest.raises(ValueError, match="at least 2"):
            traj.generate([[0.0, 0.0]], [])

    def test_duration_count_mismatch_raises(self, bk):
        """The hop-duration count must be one less than the waypoint count."""
        from shinro.trajectories.catmull_rom import CatmullRom
        traj = CatmullRom(backend=bk)
        with pytest.raises(ValueError, match="hop durations"):
            traj.generate([[0.0, 0.0], [1.0, 1.0]], [1.0, 1.0])

    def test_non_positive_duration_raises(self, bk):
        """Every hop duration must be positive."""
        from shinro.trajectories.catmull_rom import CatmullRom
        traj = CatmullRom(backend=bk)
        with pytest.raises(ValueError, match="must be positive"):
            traj.generate([[0.0, 0.0], [1.0, 1.0]], [0.0])

    def test_ragged_waypoints_raise(self, bk):
        """Waypoints of differing dimension are a loud error."""
        from shinro.trajectories.catmull_rom import CatmullRom
        traj = CatmullRom(backend=bk)
        with pytest.raises(ValueError, match="same dimension"):
            traj.generate([[0.0, 0.0], [1.0, 1.0, 1.0]], [1.0])

    def test_bad_endpoint_tangent_raises(self, bk):
        """An unknown endpoint_tangent value is a loud error."""
        from shinro.trajectories.catmull_rom import CatmullRom
        traj = CatmullRom(backend=bk)
        with pytest.raises(ValueError, match="endpoint_tangent"):
            traj.generate([[0.0, 0.0], [1.0, 1.0]], [1.0], endpoint_tangent="bogus")

    def test_from_config_schedule(self, bk):
        """from_config samples the curve into a (steps, d) waypoint schedule."""
        from shinro.trajectories.catmull_rom import CatmullRom
        schedule = CatmullRom.from_config(
            {
                "type": "catmull_rom",
                "dt": 0.5,
                "start": [0.0, 0.0],
                "waypoints": [
                    {"duration": 1.0, "position": [1.0, 1.0]},
                    {"duration": 1.0, "position": [2.0, -1.0]},
                ],
            },
            backend=bk,
        )
        arr = _to_np(schedule, bk)
        assert arr.shape == (4, 2)
        assert np.allclose(arr[0], [0.0, 0.0])


class TestDerivativeSchedules:
    """The ``derivatives`` opt-in emits stacked (pos, vel, acc) without changing the default."""

    CONFIGS = {
        "bezier": {"dt": 0.1, "duration": 1.0, "control_points": [[0, 0], [1, 1], [2, 0]]},
        "bspline": {
            "dt": 0.1,
            "duration": 1.0,
            "degree": 3,
            "control_points": [[0, 0], [0, 1], [1, 1], [1, 0]],
            "knots": [0, 0, 0, 0, 1, 1, 1, 1],
        },
        "catmull_rom": {
            "dt": 0.5,
            "start": [0, 0],
            "waypoints": [
                {"duration": 1.0, "position": [1, 1]},
                {"duration": 1.0, "position": [2, -1]},
            ],
        },
        "lissajous": {
            "dt": 0.1,
            "duration": 1.0,
            "start": [0.5, 0.5],
            "end": [-0.5, -0.5],
            "k": [1, 2],
        },
        "cubic_segments": {"dt": 0.1, "segments": [{"duration": 1.0, "start": [0, 0], "end": [1, 1]}]},
        "quintic_segments": {"dt": 0.1, "segments": [{"duration": 1.0, "start": [0, 0], "end": [1, 1]}]},
    }

    def _from_config(self, name, bk, **overrides):
        from shinro.factories.registry import _TRAJECTORY_REGISTRY
        cfg = {**self.CONFIGS[name], "type": name, **overrides}
        return _TRAJECTORY_REGISTRY[name].from_config(cfg, backend=bk)

    @pytest.mark.parametrize("name", list(CONFIGS))
    def test_default_is_positions_only(self, bk, name):
        """Without the opt-in, from_config returns the (steps, N) position schedule."""
        out = self._from_config(name, bk)
        assert not isinstance(out, dict)
        assert _to_np(out, bk).ndim == 2

    @pytest.mark.parametrize("name", list(CONFIGS))
    def test_derivatives_returns_stacked_dict(self, bk, name):
        """With derivatives=True, from_config returns position/velocity/acceleration arrays."""
        plain = _to_np(self._from_config(name, bk), bk)
        deriv = self._from_config(name, bk, derivatives=True)
        assert set(deriv) == {"position", "velocity", "acceleration"}
        for key in ("position", "velocity", "acceleration"):
            assert _to_np(deriv[key], bk).shape == plain.shape
        assert np.allclose(_to_np(deriv["position"], bk), plain)

    @pytest.mark.parametrize("name", list(CONFIGS))
    def test_velocity_matches_finite_difference(self, bk, name):
        """The emitted velocity is the finite-difference derivative of the positions."""
        dt = 0.002
        deriv = self._from_config(name, bk, dt=dt, derivatives=True)
        pos = _to_np(deriv["position"], bk)
        vel = _to_np(deriv["velocity"], bk)
        assert np.allclose(np.gradient(pos, dt, axis=0)[1:-1], vel[1:-1], atol=1e-2)


class TestSamplingHelpers:
    """Verify sample_schedule / sample_segments directly."""

    def _line(self, bk):
        from shinro.trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        traj.generate(
            bk.array([0.0, 0.0]), bk.array([1.0, 2.0]), 1.0, bk.array([0.0, 0.0]), bk.array([0.0, 0.0])
        )
        return traj

    def test_sample_schedule_orders(self, bk):
        """order selects which of position/velocity/acceleration are returned."""
        from shinro.trajectories import sample_schedule
        traj = self._line(bk)
        assert set(sample_schedule(traj, 0.1, order=0)) == {"position"}
        assert set(sample_schedule(traj, 0.1, order=1)) == {"position", "velocity"}
        out = sample_schedule(traj, 0.1, order=2)
        assert set(out) == {"position", "velocity", "acceleration"}
        assert _to_np(out["position"], bk).shape == (10, 2)

    def test_sample_schedule_reads_generator_duration(self, bk):
        """duration defaults to the generator's own horizon (CubicPolynomial stores self.duration)."""
        from shinro.trajectories import sample_schedule
        out = sample_schedule(self._line(bk), 0.1)
        assert _to_np(out["position"], bk).shape == (10, 2)

    def test_sample_segments_concatenates(self, bk):
        """Per-segment samples concatenate; the joint is emitted once."""
        from shinro.trajectories import sample_segments
        segments = []
        for a, b in (((0.0,), (1.0,)), ((1.0,), (2.0,))):
            from shinro.trajectories.cubic_polynomial import CubicPolynomial
            seg = CubicPolynomial(backend=bk)
            seg.generate(bk.array(a), bk.array(b), 1.0, bk.array([0.0]), bk.array([0.0]))
            segments.append((seg, 1.0))
        out = sample_segments(segments, 0.5, order=2)
        assert _to_np(out["position"], bk).shape == (4, 1)
        assert np.allclose(_to_np(out["position"], bk)[2], 1.0)


def _make_waypoint_spline(cls, bk, start, wps, durations):
    traj = cls(backend=bk)
    traj.generate([start, *wps], list(durations))
    return traj


class TestAkima:
    """Verify Akima: C1 pass-through, overshoot resistance, validation."""

    POINTS = ((0.0, 0.0), (1.0, 1.0), (2.0, -1.0), (3.0, 1.0))
    DURATIONS = (1.0, 1.0, 1.0)

    def _make(self, bk):
        from shinro.trajectories.waypoint_splines import Akima
        return _make_waypoint_spline(Akima, bk, self.POINTS[0], self.POINTS[1:], self.DURATIONS)

    def test_passes_through_waypoints(self, bk):
        """The curve hits the start and every waypoint at its cumulative time."""
        traj = self._make(bk)
        assert np.allclose(_to_np(traj.position_at(0.0)[0], bk), self.POINTS[0])
        t = 0.0
        for i, d in enumerate(self.DURATIONS):
            t += d
            assert np.allclose(_to_np(traj.position_at(t)[0], bk), self.POINTS[i + 1])

    def test_velocity_continuous_at_interior_waypoints(self, bk):
        """Left/right velocities agree at every interior waypoint (C1)."""
        traj = self._make(bk)
        segs = [seg for (_, _, seg) in traj._segments]
        for i in range(len(segs) - 1):
            v_left = _to_np(segs[i].position_at(self.DURATIONS[i])[1], bk)
            v_right = _to_np(segs[i + 1].position_at(0.0)[1], bk)
            assert np.allclose(v_left, v_right)

    def test_does_not_overshoot_step(self, bk):
        """Akima stays inside the data range across a flat-then-step change."""
        from shinro.trajectories.waypoint_splines import Akima
        pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]
        traj = _make_waypoint_spline(Akima, bk, pts[0], pts[1:], [1.0] * 5)
        ys = np.array([float(_to_np(traj.position_at(t)[0], bk)[1]) for t in np.linspace(0.0, 5.0, 501)])
        assert ys.min() >= -1e-9
        assert ys.max() <= 1.0 + 1e-9

    def test_single_hop_is_linear(self, bk):
        """With one interval there is no four-slope stencil — fall back to a line."""
        from shinro.trajectories.waypoint_splines import Akima
        traj = _make_waypoint_spline(Akima, bk, [0.0, 0.0], [[1.0, 1.0]], [1.0])
        for t in (0.0, 0.25, 0.5, 1.0):
            assert np.allclose(_to_np(traj.position_at(t)[0], bk), [t, t])
        assert np.allclose(_to_np(traj.position_at(0.5)[2], bk), 0.0)

    def test_ndimensional(self, bk):
        from shinro.trajectories.waypoint_splines import Akima
        traj = _make_waypoint_spline(Akima, bk, [0.0, 0.0, 0.0], [[1.0, 1.0, 1.0], [2.0, 0.0, 2.0]], [1.0, 1.0])
        assert _to_np(traj.position_at(1.0)[0], bk).shape == (3,)

    def test_validation(self, bk):
        from shinro.trajectories.waypoint_splines import Akima
        traj = Akima(backend=bk)
        with pytest.raises(ValueError, match="at least 2"):
            traj.generate([[0.0, 0.0]], [])
        with pytest.raises(ValueError, match="hop durations"):
            traj.generate([[0.0, 0.0], [1.0, 1.0]], [1.0, 1.0])
        with pytest.raises(ValueError, match="must be positive"):
            traj.generate([[0.0, 0.0], [1.0, 1.0]], [0.0])
        with pytest.raises(ValueError, match="same dimension"):
            traj.generate([[0.0, 0.0], [1.0, 1.0, 1.0]], [1.0])

    def test_from_config_schedule_and_derivatives(self, bk):
        """from_config returns positions (and the derivative dict when opted in)."""
        from shinro.trajectories.waypoint_splines import Akima
        cfg = {
            "type": "akima",
            "dt": 0.5,
            "start": [0.0, 0.0],
            "waypoints": [
                {"duration": 1.0, "position": [1.0, 1.0]},
                {"duration": 1.0, "position": [2.0, -1.0]},
            ],
        }
        arr = _to_np(Akima.from_config(cfg, backend=bk), bk)
        assert arr.shape == (4, 2)
        deriv = Akima.from_config({**cfg, "derivatives": True}, backend=bk)
        assert set(deriv) == {"position", "velocity", "acceleration"}
        assert np.allclose(_to_np(deriv["position"], bk), arr)


class TestCubicSpline:
    """Verify CubicSpline: C2 pass-through, natural end conditions, validation."""

    POINTS = ((0.0, 0.0), (1.0, 1.0), (2.0, -1.0), (3.0, 1.0))
    DURATIONS = (1.0, 1.0, 1.0)

    def _make(self, bk):
        from shinro.trajectories.waypoint_splines import CubicSpline
        return _make_waypoint_spline(CubicSpline, bk, self.POINTS[0], self.POINTS[1:], self.DURATIONS)

    def test_passes_through_waypoints(self, bk):
        traj = self._make(bk)
        assert np.allclose(_to_np(traj.position_at(0.0)[0], bk), self.POINTS[0])
        t = 0.0
        for i, d in enumerate(self.DURATIONS):
            t += d
            assert np.allclose(_to_np(traj.position_at(t)[0], bk), self.POINTS[i + 1])

    def test_velocity_continuous_at_interior_waypoints(self, bk):
        """C1: left/right velocities agree at interior waypoints."""
        traj = self._make(bk)
        segs = [seg for (_, _, seg) in traj._segments]
        for i in range(len(segs) - 1):
            assert np.allclose(
                _to_np(segs[i].position_at(self.DURATIONS[i])[1], bk),
                _to_np(segs[i + 1].position_at(0.0)[1], bk),
            )

    def test_acceleration_continuous_at_interior_waypoints(self, bk):
        """C2: left/right accelerations agree at interior waypoints."""
        traj = self._make(bk)
        segs = [seg for (_, _, seg) in traj._segments]
        for i in range(len(segs) - 1):
            assert np.allclose(
                _to_np(segs[i].position_at(self.DURATIONS[i])[2], bk),
                _to_np(segs[i + 1].position_at(0.0)[2], bk),
                atol=1e-9,
            )

    def test_natural_end_acceleration_zero(self, bk):
        """The natural conditions pin boundary acceleration to zero."""
        traj = self._make(bk)
        assert np.allclose(_to_np(traj.position_at(0.0)[2], bk), 0.0, atol=1e-9)
        assert np.allclose(_to_np(traj.position_at(traj.T)[2], bk), 0.0, atol=1e-9)

    def test_natural_step_overshoots(self, bk):
        """Contrast with Akima: the global natural spline overshoots a step."""
        from shinro.trajectories.waypoint_splines import CubicSpline
        pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]
        traj = _make_waypoint_spline(CubicSpline, bk, pts[0], pts[1:], [1.0] * 5)
        ys = np.array([float(_to_np(traj.position_at(t)[0], bk)[1]) for t in np.linspace(0.0, 5.0, 501)])
        assert ys.min() < -1e-3 or ys.max() > 1.0 + 1e-3

    def test_ndimensional(self, bk):
        from shinro.trajectories.waypoint_splines import CubicSpline
        traj = _make_waypoint_spline(CubicSpline, bk, [0.0, 0.0, 0.0], [[1.0, 1.0, 1.0], [2.0, 0.0, 2.0]], [1.0, 1.0])
        assert _to_np(traj.position_at(1.0)[0], bk).shape == (3,)

    def test_from_config_schedule_and_derivatives(self, bk):
        from shinro.trajectories.waypoint_splines import CubicSpline
        cfg = {
            "type": "cubic_spline",
            "dt": 0.5,
            "start": [0.0, 0.0],
            "waypoints": [
                {"duration": 1.0, "position": [1.0, 1.0]},
                {"duration": 1.0, "position": [2.0, -1.0]},
            ],
        }
        arr = _to_np(CubicSpline.from_config(cfg, backend=bk), bk)
        assert arr.shape == (4, 2)
        deriv = CubicSpline.from_config({**cfg, "derivatives": True}, backend=bk)
        assert set(deriv) == {"position", "velocity", "acceleration"}
        assert np.allclose(_to_np(deriv["position"], bk), arr)


class TestMinSnap:
    """Verify MinSnapPolynomial: rest-to-rest closed form, boundary conditions."""

    def _make(self, bk, p0, pf, T, **boundaries):
        from shinro.trajectories.min_snap import MinSnapPolynomial
        traj = MinSnapPolynomial(backend=bk)
        traj.generate(bk.array(p0), bk.array(pf), T, **boundaries)
        return traj

    def test_rest_to_rest_matches_closed_form(self, bk):
        """All-zero boundaries reduce to p0 + (pf-p0)(35s^4 - 84s^5 + 70s^6 - 20s^7)."""
        p0, pf, T = [0.0, 1.0], [2.0, -1.0], 2.0
        traj = self._make(bk, p0, pf, T)
        for t in (0.0, 0.4, 1.0, 1.6, 2.0):
            s = t / T
            want = np.array(p0) + (np.array(pf) - np.array(p0)) * (
                35 * s**4 - 84 * s**5 + 70 * s**6 - 20 * s**7
            )
            assert np.allclose(_to_np(traj.position_at(t)[0], bk), want)

    def test_endpoints_interpolated(self, bk):
        traj = self._make(bk, [0.0, 0.0], [1.0, 2.0], 1.5)
        assert np.allclose(_to_np(traj.position_at(0.0)[0], bk), [0.0, 0.0])
        assert np.allclose(_to_np(traj.position_at(1.5)[0], bk), [1.0, 2.0])

    def test_rest_to_rest_zero_boundary_velocity_acceleration(self, bk):
        """Zero boundary derivatives pin v and a to zero at both ends."""
        traj = self._make(bk, [0.0], [1.0], 2.0)
        for t, which in ((0.0, 1), (2.0, 1), (0.0, 2), (2.0, 2)):
            assert np.allclose(_to_np(traj.position_at(t)[which], bk), 0.0, atol=1e-9)

    def test_nonzero_boundary_velocity(self, bk):
        """A supplied start velocity is honoured without moving the endpoints."""
        traj = self._make(bk, [0.0], [1.0], 2.0, start_vel=bk.array([0.5]))
        assert np.allclose(_to_np(traj.position_at(0.0)[1], bk), 0.5, atol=1e-9)
        assert np.allclose(_to_np(traj.position_at(0.0)[0], bk), 0.0)
        assert np.allclose(_to_np(traj.position_at(2.0)[0], bk), 1.0)

    def test_derivative_of_position_is_velocity(self, bk):
        traj = self._make(bk, [0.0], [1.0], 2.0)
        eps = 1e-6
        p_plus = _to_np(traj.position_at(0.8 + eps)[0], bk)
        p_minus = _to_np(traj.position_at(0.8 - eps)[0], bk)
        assert np.allclose((p_plus - p_minus) / (2 * eps), _to_np(traj.position_at(0.8)[1], bk), atol=1e-4)

    def test_bad_duration_raises(self, bk):
        from shinro.trajectories.min_snap import MinSnapPolynomial
        traj = MinSnapPolynomial(backend=bk)
        with pytest.raises(ValueError, match="duration must be positive"):
            traj.generate(bk.array([0.0]), bk.array([1.0]), 0.0)

    def test_from_config_schedule_and_derivatives(self, bk):
        from shinro.trajectories.min_snap import MinSnapPolynomial
        cfg = {"type": "min_snap", "dt": 0.5, "start": [0.0, 0.0], "end": [1.0, 1.0], "duration": 1.0}
        arr = _to_np(MinSnapPolynomial.from_config(cfg, backend=bk), bk)
        assert arr.shape == (2, 2)
        deriv = MinSnapPolynomial.from_config({**cfg, "derivatives": True}, backend=bk)
        assert set(deriv) == {"position", "velocity", "acceleration"}
        assert np.allclose(_to_np(deriv["position"], bk), arr)


class TestCircularArc:
    """Verify CircularArc: constant radius, analytic speed/accel, helix, validation."""

    def _make(self, bk, center, start, T, sweep, normal=None, pitch=0.0):
        from shinro.trajectories.circular_arc import CircularArc
        traj = CircularArc(backend=bk)
        traj.generate(center, start, T, sweep, normal=normal, pitch=pitch)
        return traj

    def test_radius_constant(self, bk):
        traj = self._make(bk, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0, np.pi / 2)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            pos = _to_np(traj.position_at(t)[0], bk)
            assert np.isclose(np.linalg.norm(pos), 0.5)

    def test_endpoints_quarter_turn(self, bk):
        traj = self._make(bk, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0, np.pi / 2)
        assert np.allclose(_to_np(traj.position_at(0.0)[0], bk), [0.5, 0.0, 0.0])
        assert np.allclose(_to_np(traj.position_at(1.0)[0], bk), [0.0, 0.5, 0.0], atol=1e-12)

    def test_full_turn_returns_to_start(self, bk):
        traj = self._make(bk, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0, 2 * np.pi)
        assert np.allclose(_to_np(traj.position_at(1.0)[0], bk), [0.5, 0.0, 0.0], atol=1e-12)

    def test_constant_speed_and_centripetal_acceleration(self, bk):
        """Speed is r*|omega| and acceleration points at the centre (r*omega^2)."""
        radius, sweep, T = 0.5, np.pi / 2, 1.0
        omega = sweep / T
        traj = self._make(bk, [0.0, 0.0, 0.0], [radius, 0.0, 0.0], T, sweep)
        for t in (0.2, 0.5, 0.8):
            pos = _to_np(traj.position_at(t)[0], bk)
            vel = _to_np(traj.position_at(t)[1], bk)
            acc = _to_np(traj.position_at(t)[2], bk)
            assert np.isclose(np.linalg.norm(vel), radius * omega)
            assert np.isclose(np.linalg.norm(acc), radius * omega**2)
            assert np.allclose(acc / np.linalg.norm(acc), -pos / np.linalg.norm(pos))

    def test_tilted_plane(self, bk):
        """A non-default normal tilts the arc plane; the radius is preserved."""
        traj = self._make(
            bk, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0, 2 * np.pi, normal=[0.0, 1.0, 0.0]
        )
        # rotating about +y takes (0.5,0,0) out of the xy-plane
        end = _to_np(traj.position_at(1.0)[0], bk)
        assert np.isclose(np.linalg.norm(end), 0.5)
        assert np.allclose(end, [0.5, 0.0, 0.0], atol=1e-12)

    def test_helix_pitch(self, bk):
        traj = self._make(bk, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0, 2 * np.pi, pitch=0.1)
        end = _to_np(traj.position_at(1.0)[0], bk) - _to_np(traj.position_at(0.0)[0], bk)
        assert np.isclose(end[2], 0.1)

    def test_validation(self, bk):
        from shinro.trajectories.circular_arc import CircularArc
        traj = CircularArc(backend=bk)
        with pytest.raises(ValueError, match="duration must be positive"):
            traj.generate([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 0.0, 1.0)
        with pytest.raises(ValueError, match="3-D"):
            traj.generate([0.0, 0.0], [0.5, 0.0], 1.0, 1.0)
        with pytest.raises(ValueError, match="non-zero"):
            traj.generate([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0, 1.0, normal=[0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="differ"):
            traj.generate([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 1.0)
        with pytest.raises(ValueError, match="perpendicular"):
            traj.generate([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0, 1.0, normal=[1.0, 0.0, 0.0])

    def test_from_config_schedule_and_derivatives(self, bk):
        from shinro.trajectories.circular_arc import CircularArc
        cfg = {
            "type": "circular_arc",
            "dt": 0.1,
            "center": [0.0, 0.0, 0.0],
            "start": [0.5, 0.0, 0.0],
            "duration": 1.0,
            "sweep_angle": np.pi / 2,
        }
        arr = _to_np(CircularArc.from_config(cfg, backend=bk), bk)
        assert arr.shape == (10, 3)
        deriv = CircularArc.from_config({**cfg, "derivatives": True}, backend=bk)
        assert set(deriv) == {"position", "velocity", "acceleration"}
        assert np.allclose(_to_np(deriv["position"], bk), arr)
