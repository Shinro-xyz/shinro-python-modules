"""Shared helpers for demo scripts and integration tests."""

from pathlib import Path

import mujoco
import numpy as np


def load_model_assets(mesh_dir: Path) -> dict:
    """Load mesh files from a directory for XML injection."""
    assets = {}
    if mesh_dir.exists():
        for fname in mesh_dir.iterdir():
            if fname.suffix in ('.stl', '.obj'):
                assets[fname.name] = fname.read_bytes()
    return assets


def inject_free_joint(xml_string: str) -> str:
    """Nest the arm base body under the wheel base and insert a free joint.

    The stock LeKiwi MJCF has the arm as a sibling of the wheel base. This
    rewrites the tree so the arm hangs off the (mobile) base plate and the
    base gains a free joint, enabling base mobility in MuJoCo. The XML is
    round-tripped through ``xml.etree.ElementTree``.

    Args:
        xml_string: Original MJCF as a string.

    Returns:
        Rewritten MJCF as a string.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_string)
    worldbody = root.find('.//worldbody')
    wheel_base = arm_base = None
    for child in list(worldbody):
        if child.tag == 'body':
            name = child.get('name', '')
            if 'base_plate_layer1' in name:
                wheel_base = child
            elif 'base_plate_layer2' in name:
                arm_base = child
    if wheel_base is not None and arm_base is not None:
        wb_pos = [float(x) for x in wheel_base.get('pos', '0 0 0').split()]
        ab_pos = [float(x) for x in arm_base.get('pos', '0 0 0').split()]
        rel_pos = [ab_pos[i] - wb_pos[i] for i in range(3)]
        arm_base.set('pos', f'{rel_pos[0]} {rel_pos[1]} {rel_pos[2]}')
        worldbody.remove(arm_base)
        fj = ET.Element('freejoint')
        wheel_base.insert(0, fj)
        wheel_base.append(arm_base)
    return ET.tostring(root, encoding='unicode')


def load_lekiwi_mjcf(mjcf_path: Path) -> str:
    """Read the LeKiwi MJCF XML from disk.

    Args:
        mjcf_path: Path to the MJCF file.

    Returns:
        The XML document as a string.
    """
    return mjcf_path.read_text()


def setup_gif_renderer(engine, width=640, height=400):
    """Create a MuJoCo renderer and camera for GIF capture."""
    renderer = mujoco.Renderer(engine.model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.distance = 1.5
    camera.azimuth = 135
    camera.elevation = -20
    camera.lookat[:] = [0.0, 0.0, 0.1]
    return renderer, camera


def setup_live_viewer(engine, distance=2.0, azimuth=90, elevation=-30, lookat=(0.0, 0.0, 0.1)):
    """Open a live MuJoCo viewer."""
    viewer = mujoco.viewer.launch_passive(engine.model, engine.data)
    viewer.cam.distance = distance
    viewer.cam.azimuth = azimuth
    viewer.cam.elevation = elevation
    viewer.cam.lookat[:] = lookat
    return viewer


def save_gif(frames, path, fps=12):
    """Save frames as a GIF."""
    import imageio
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"✅ GIF saved: {path} ({len(frames)} frames, {fps} fps)")


def setup_dark_plot(fig, axes):
    """Apply dark theme to a matplotlib figure."""
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white', labelsize=7)
        ax.spines['bottom'].set_color('#555')
        ax.spines['top'].set_color('#555')
        ax.spines['left'].set_color('#555')
        ax.spines['right'].set_color('#555')


def make_composite_frame(mujoco_frame, fig):
    """Composite a MuJoCo render next to a matplotlib plot."""
    from io import BytesIO

    from PIL import Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plot_pil = Image.open(buf)
    plot_frame = np.array(plot_pil.convert('RGB'))

    mj_h, mj_w = mujoco_frame.shape[:2]
    plot_h, plot_w = plot_frame.shape[:2]
    plot_pil = Image.fromarray(plot_frame)
    plot_pil = plot_pil.resize((int(plot_pil.width * mj_h / plot_h), mj_h), Image.Resampling.LANCZOS)
    return np.hstack([mujoco_frame, np.array(plot_pil)])
