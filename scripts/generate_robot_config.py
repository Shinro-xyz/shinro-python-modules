#!/usr/bin/env python3
"""
Generate a robot_config.toml from a MuJoCo XML model file.

Auto-detects plant types using the detector registry. Supports single plants,
combined robots (e.g., arm + base), and unknown XMLs with a fallback.

Usage:
    python scripts/generate_robot_config.py models/pendulum.xml
    python scripts/generate_robot_config.py lekiwi-sim/mjcf_lcmm_robot.xml
    python scripts/generate_robot_config.py models/ --output-dir configs/plants/
    python scripts/generate_robot_config.py some_robot.xml --type InvertedPendulum
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from shinro.factories.registry import _PLANT_DETECTOR_REGISTRY


def _parse_pos(pos_str: str) -> np.ndarray:
    return np.array([float(x) for x in pos_str.split()]) if pos_str else np.zeros(3)


def _parse_euler(euler_str: str) -> np.ndarray:
    return np.array([float(x) for x in euler_str.split()]) if euler_str else np.zeros(3)


def _parse_quat(quat_str: str) -> np.ndarray:
    return np.array([float(x) for x in quat_str.split()]) if quat_str else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def _euler_to_rotmat(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _body_transform(body: ET.Element, parent_transform: np.ndarray = np.eye(4)) -> np.ndarray:
    pos = _parse_pos(body.get('pos'))
    euler = _parse_euler(body.get('euler'))
    quat = _parse_quat(body.get('quat'))
    T = np.eye(4)
    if body.get('quat') is not None:
        T[:3, :3] = _quat_to_rotmat(quat)
    else:
        T[:3, :3] = _euler_to_rotmat(euler)
    T[:3, 3] = pos
    return parent_transform @ T


def _axis_from_defaults(defaults: list, class_name: str) -> str | None:
    for d in defaults:
        if d.get('class') == class_name:
            joint_elem = d.find('joint')
            if joint_elem is not None and joint_elem.get('axis') is not None:
                return joint_elem.get('axis')
    return None


def _axis_to_letter(axis_str: str) -> str:
    vals = np.array([float(x) for x in axis_str.split()])
    return ['x', 'y', 'z'][np.argmax(np.abs(vals))]


def _build_actuator_map(root: ET.Element) -> dict[str, str]:
    actuator_map = {}
    for act in root.findall('.//actuator/*'):
        jname = act.get('joint')
        if jname:
            actuator_map[jname] = act.tag
    return actuator_map


def _find_arm_chain(root: ET.Element, actuator_map: dict[str, str]) -> list[tuple[str, ET.Element, ET.Element, np.ndarray]]:
    worldbody = root.find('.//worldbody')

    def walk(body_elem, parent_T=np.eye(4)):
        results = []
        T_body = _body_transform(body_elem, parent_T)
        joint = body_elem.find('joint')
        if joint is not None:
            jname = joint.get('name')
            if jname and jname in actuator_map and actuator_map[jname] == 'position':
                jpos = _parse_pos(joint.get('pos'))
                T_joint = T_body.copy()
                T_joint[:3, 3] = T_body[:3, :3] @ jpos + T_body[:3, 3]
                results.append((jname, joint, body_elem, T_joint))
        for child in body_elem.findall('body'):
            results.extend(walk(child, T_body))
        return results

    return walk(worldbody)


def _compute_link_offsets(arm_chain: list) -> list[list[float]]:
    offsets = []
    for i in range(len(arm_chain)):
        if i + 1 < len(arm_chain):
            R_i = arm_chain[i][3][:3, :3]
            offset = R_i.T @ (arm_chain[i + 1][3][:3, 3] - arm_chain[i][3][:3, 3])
            offsets.append([round(v, 6) for v in offset.tolist()])
        else:
            offsets.append([0.0, 0.0, 0.0])
    return offsets


def _extract_physical_params(root: ET.Element) -> dict:
    """Extract physical parameters from the XML (masses, inertias, joint limits, etc.)."""
    params = {}
    bodies = root.findall('.//body')
    for body in bodies:
        inertial = body.find('inertial')
        if inertial is not None:
            mass = inertial.get('mass')
            if mass:
                params.setdefault('masses', {})[body.get('name', 'unknown')] = float(mass)
        geoms = body.findall('geom')
        for geom in geoms:
            mass = geom.get('mass')
            if mass:
                params.setdefault('masses', {})[geom.get('name', body.get('name', 'unknown'))] = float(mass)
    joints = root.findall('.//joint')
    for joint in joints:
        jname = joint.get('name', 'unknown')
        jrange = joint.get('range')
        if jrange:
            params.setdefault('joint_ranges', {})[jname] = [float(x) for x in jrange.split()]
        damping = joint.get('damping')
        if damping:
            params.setdefault('damping', {})[jname] = float(damping)
    actuators = root.findall('.//actuator/*')
    for act in actuators:
        aname = act.get('name', act.get('joint', 'unknown'))
        ctrlrange = act.get('ctrlrange')
        if ctrlrange:
            params.setdefault('actuator_ranges', {})[aname] = [float(x) for x in ctrlrange.split()]
    option = root.find('.//option')
    if option is not None:
        gravity = option.get('gravity')
        if gravity:
            params['gravity'] = [float(x) for x in gravity.split()]
    return params


def _generate_armrobot_config(root: ET.Element, xml_path: str) -> dict:
    """Generate ArmRobot config from XML."""
    actuator_map = _build_actuator_map(root)
    arm_joints = [j for j, t in actuator_map.items() if t == 'position']
    arm_chain = _find_arm_chain(root, actuator_map)
    defaults = root.findall('.//default')

    rot_axes = []
    for _, joint_elem, _, _ in arm_chain:
        axis = joint_elem.get('axis')
        if axis is None:
            class_name = joint_elem.get('class')
            if class_name:
                axis = _axis_from_defaults(defaults, class_name)
        if axis is None:
            axis = "0 0 1"
        rot_axes.append(_axis_to_letter(axis))

    link_offsets = _compute_link_offsets(arm_chain)
    ee_body_name = arm_chain[-1][2].get('name', '') if arm_chain else ""

    return {
        "type": "ArmRobot", "name": "arm", "num_dof": len(arm_joints),
        "joint_group": "arm_joints", "ee_body_name": ee_body_name,
        "rot_axes": rot_axes, "joint_offsets": link_offsets,
    }


def _generate_holonomic_config(root: ET.Element, xml_path: str) -> dict:
    """Generate HolonomicMobileRobot config from XML."""
    actuator_map = _build_actuator_map(root)
    drive_joints = [j for j, t in actuator_map.items() if t == 'motor']
    return {
        "type": "HolonomicMobileRobot", "name": "base",
        "num_wheels": len(drive_joints), "radius_robots": 0.12,
        "gamma": -1.57079632679, "radius_wheels": 0.09,
    }


def _generate_pendulum_config(root: ET.Element, xml_path: str) -> dict:
    """Generate InvertedPendulum config from XML."""
    params = _extract_physical_params(root)
    masses = list(params.get('masses', {}).values())
    total_mass = sum(masses) if masses else 0.1
    damping = list(params.get('damping', {}).values())
    return {
        "type": "InvertedPendulum", "name": "pendulum",
        "mass": total_mass, "length": 0.5,
        "damping": damping[0] if damping else 0.0,
        "gravity": abs(params.get('gravity', [0, 0, -9.81])[2]),
        "dt": 0.01,
    }


def _generate_cartpole_config(root: ET.Element, xml_path: str) -> dict:
    """Generate CartPole config from XML."""
    params = _extract_physical_params(root)
    masses = list(params.get('masses', {}).values())
    cart_mass = masses[0] if len(masses) > 0 else 0.5
    pole_mass = masses[1] if len(masses) > 1 else 0.1
    damping = list(params.get('damping', {}).values())
    joint_ranges = list(params.get('joint_ranges', {}).values())
    track_limit = joint_ranges[0] if joint_ranges else [-2.0, 2.0]
    return {
        "type": "CartPole", "name": "cartpole",
        "cart_mass": cart_mass, "pole_mass": pole_mass, "pole_length": 0.5,
        "damping": damping[0] if damping else 0.0,
        "gravity": abs(params.get('gravity', [0, 0, -9.81])[2]),
        "dt": 0.01, "track_limits": track_limit,
    }


_PLANT_CONFIG_GENERATORS = {
    "ArmRobot": _generate_armrobot_config,
    "HolonomicMobileRobot": _generate_holonomic_config,
    "InvertedPendulum": _generate_pendulum_config,
    "CartPole": _generate_cartpole_config,
}


def detect_plant_types(root: ET.Element, cli_type: str = None) -> list[str]:
    """Detect plant types from XML using three-tier fallback.

    1. CLI --type flag (highest priority)
    2. XML annotation (<plant type="..."/>)
    3. Heuristic match from detector registry
    """
    if cli_type:
        return [cli_type]

    annotation = root.find('.//plant')
    if annotation is not None:
        ptype = annotation.get('type')
        if ptype:
            return [ptype]

    detected = []
    for plant_type, detector in _PLANT_DETECTOR_REGISTRY.items():
        if detector(root):
            detected.append(plant_type)
    return detected


def generate_config(xml_path: str, cli_type: str = None) -> dict:
    """Generate a complete TOML config dict from an XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    config = {"model": str(Path(xml_path).as_posix()), "dt": 0.02}

    plant_types = detect_plant_types(root, cli_type)

    if not plant_types:
        print(f"Warning: no plant type detected for {xml_path}. "
              "Use --type flag or add <plant type=\"...\"/> to the XML.", file=sys.stderr)
        return config

    actuator_map = _build_actuator_map(root)
    arm_joints = [j for j, t in actuator_map.items() if t == 'position']
    drive_joints = [j for j, t in actuator_map.items() if t == 'motor']

    joint_groups = {}
    if arm_joints:
        joint_groups["arm_joints"] = arm_joints
    if drive_joints:
        joint_groups["drive_joints"] = drive_joints
    if joint_groups:
        config["joint_groups"] = joint_groups

    plants = []
    for ptype in plant_types:
        generator = _PLANT_CONFIG_GENERATORS.get(ptype)
        if generator:
            plants.append(generator(root, xml_path))
        else:
            print(f"Warning: no config generator for plant type '{ptype}'", file=sys.stderr)
    if plants:
        config["plants"] = plants

    return config


def _format_toml_list(items: list, indent: int = 0) -> str:
    if len(items) == 0:
        return "[]"
    if isinstance(items[0], (int, float)):
        inner = ", ".join(str(v) for v in items)
        return f"[{inner}]"
    if isinstance(items[0], list):
        inner = ", ".join(_format_toml_list(v) for v in items)
        return f"[{inner}]"
    return "[]"


def _format_toml_value(v) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return _format_toml_list(v)
    return str(v)


def toml_string(config: dict) -> str:
    """Convert a config dict to a TOML string."""
    lines = []
    for key in ["model", "dt"]:
        if key in config:
            lines.append(f'{key} = {_format_toml_value(config[key])}')
    lines.append("")

    jg = config.get("joint_groups", {})
    if jg:
        lines.append("[joint_groups]")
        for name, joints in jg.items():
            items = ", ".join(f'"{j}"' for j in joints)
            lines.append(f'{name} = [{items}]')
        lines.append("")

    plants = config.get("plants", [])
    for p in plants:
        lines.append("[[plants]]")
        for key in ["type", "name", "num_dof", "joint_group", "ee_body_name", "num_wheels",
                    "radius_robots", "gamma", "radius_wheels", "mass", "length", "damping",
                    "gravity", "cart_mass", "pole_mass", "pole_length"]:
            if key in p:
                lines.append(f'{key} = {_format_toml_value(p[key])}')
        if "rot_axes" in p:
            items = ", ".join(f'"{a}"' for a in p["rot_axes"])
            lines.append(f'rot_axes = [{items}]')
        if "joint_offsets" in p:
            lines.append("joint_offsets = [")
            for row in p["joint_offsets"]:
                inner = ", ".join(str(v) for v in row)
                lines.append(f"  [{inner}],")
            lines.append("]")
        if "track_limits" in p:
            inner = ", ".join(str(v) for v in p["track_limits"])
            lines.append(f'track_limits = [{inner}]')
        lines.append("")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate robot_config.toml from MuJoCo XML")
    parser.add_argument("input", help="XML file path or directory (with --output-dir)")
    parser.add_argument("--type", help="Override plant type detection")
    parser.add_argument("--output-dir", help="Output directory for batch mode")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        if not args.output_dir:
            print("Error: --output-dir is required for directory input", file=sys.stderr)
            sys.exit(1)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for xml_file in sorted(input_path.glob("*.xml")):
            config = generate_config(str(xml_file), args.type)
            if config.get("plants"):
                out_name = xml_file.stem + ".toml"
                out_path = output_dir / out_name
                with open(out_path, "w") as f:
                    f.write(toml_string(config))
                print(f"Generated {out_path}")
            else:
                print(f"Skipped {xml_file.name} — no plants detected")
    else:
        if not input_path.exists():
            print(f"Error: file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        config = generate_config(str(input_path), args.type)
        print(toml_string(config))


if __name__ == "__main__":
    main()
