"""Full-loop integration tests (Trajectory → Controller → Estimator → Plant → Engine).

This suite requires MuJoCo and the LeKiwi assets; it is skipped where they are
unavailable and excluded from the default test run via the ``integration``
pytest marker. Run it with ``make test-integration``.
"""
