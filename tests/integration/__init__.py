"""Full-loop integration tests (Trajectory → Controller → Estimator → Plant → Engine).

This suite requires MuJoCo; it is skipped where it is unavailable and excluded
from the default test run via the ``integration`` pytest marker. Run it with
``make test-integration``.
"""
