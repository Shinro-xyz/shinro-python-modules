"""
Double pendulum balance demo — terminal output, no MuJoCo required.

Demonstrates the intended usage: one scenario TOML, three lines of Python.

Usage:
  python -m demos.demo_double_pendulum                 # print state
  python -m demos.demo_double_pendulum --plot          # matplotlib plot
"""
import sys

PLOT = "--plot" in sys.argv

from shinro import ScenarioFactory

SCENARIO = "configs/scenarios/double_pendulum_balance.toml"

scenario = ScenarioFactory(SCENARIO).build()
print(f"Scenario: {scenario.config.get('scenario', {}).get('name', 'unnamed')}")
print(f"Plant:    {type(scenario.plant).__name__}")
print(f"Controller: {type(scenario.controller).__name__}")
print(f"Estimator: {type(scenario.estimator).__name__}")
print()

result = scenario.run()
print(f"Ran {len(result)} steps")

# Print checkpoints at ~0s, 5s, 10s, 15s, 20s
for r in result:
    if int(r.t) in (0, 5, 10, 15, 19) and abs(r.t - int(r.t)) < 0.005:
        print(f"  t={r.t:5.1f}s: θ1={r.true_state[0]:+.4f}  θ2={r.true_state[1]:+.4f}  "
              f"ω1={r.true_state[2]:+.4f}  ω2={r.true_state[3]:+.4f}")

final = result[-1]
print(f"\nFinal state:")
print(f"  θ1 = {final.true_state[0]:+.6f} rad")
print(f"  θ2 = {final.true_state[1]:+.6f} rad")
print(f"  |θ1| < 0.01: {abs(final.true_state[0]) < 0.01}")
print(f"  |θ2| < 0.01: {abs(final.true_state[1]) < 0.01}")

if PLOT:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.array([r.t for r in result])
    states = np.array([r.true_state for r in result])
    controls = np.array([r.control for r in result])
    estimated = np.array([r.estimated for r in result])

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white', labelsize=7)
        ax.spines['bottom'].set_color('#555')
        ax.spines['top'].set_color('#555')
        ax.spines['left'].actualset_color('#555')
        ax.spines['right'].set_color('#555')

    ax = axes[0]
    ax.set_title('Joint angles (true)', color='white', fontsize=9, fontweight='bold')
    ax.set_ylabel('Angle (rad)', color='white')
    ax.plot(t, states[:, 0], '#ff6b6b', lw=1.0, label='θ1 (top)')
    ax.plot(t, states[:, 1], '#4ecdc4', lw=1.0, label='θ2 (bottom)')
    ax.legend(loc='upper right', fontsize=7, labelcolor='white', framealpha=0.3)

    ax = axes[1]
    ax.set_title('True vs estimated state', color='white', fontsize=9, fontweight='bold')
    ax.set_ylabel('Angle (rad)', color='white')
    ax.plot(t, states[:, 0], '#ff6b6b', lw=1.0, alpha=0.3, label='θ1 true')
    ax.plot(t, estimated[:, 0], '#ff6b6b', lw=1.0, ls='--', label='θ1 est')
    ax.plot(t, states[:, 1], '#4ecdc4', lw=1.0, alpha=0.3, label='θ2 true')
    ax.plot(t, estimated[:, 1], '#4ecdc4', lw=1.0, ls='--', label='θ2 est')
    ax.legend(loc='upper right', fontsize=6, labelcolor='white', framealpha=0.3, ncol=2)

    ax = axes[2]
    ax.set_title('Control effort (torques)', color='white', fontsize=9, fontweight='bold')
    ax.set_ylabel('Torque (N·m)', color='white')
    ax.set_xlabel('Time (s)', color='white')
    ax.plot(t, controls[:, 0], '#45b7d1', lw=1.0, label='τ1')
    ax.plot(t, controls[:, 1], '#ffe66d', lw=1.0, label='τ2')
    ax.legend(loc='upper right', fontsize=6, labelcolor='white', framealpha=0.3)

    plt.tight_layout()
    plt.savefig('double_pendulum_demo.png', dpi=150, facecolor=fig.get_facecolor())
    print("Plot saved: double_pendulum_demo.png")

print("Done.")
