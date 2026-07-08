#!/usr/bin/env python3
"""
Generate a robot_config.yaml from a MuJoCo XML model file.

Usage:
    python scripts/generate_robot_config.py lekiwi-sim/mjcf_lcmm_robot.xml > robot_config.yaml
    python scripts/generate_robot_config.py lekiwi-sim/so_arm100.xml > so_arm100_config.yaml

The script:
  1. Parses the XML body tree and actuator definitions
  2. Classifies joints: <position> actuators → arm, <motor> actuators → drive
  3. Computes link offsets from body positions along the arm chain
  4. Extracts rotation axes from joint/default class definitions
  5. Detects the end-effector body (last body in the arm chain)
  6. Outputs a complete YAML config
"""

import sys
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from pathlib import Path


class RobotConfigGenerator:
    """Generates a robot_config.yaml from a MuJoCo XML model file."""

    def __init__(self, xml_path: str):
        self.xml_path = Path(xml_path)
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        self.defaults = self.root.findall('.//default')

    def _parse_pos(self, pos_str: str) -> np.ndarray:
        return np.array([float(x) for x in pos_str.split()]) if pos_str else np.zeros(3)

    def _parse_euler(self, euler_str: str) -> np.ndarray:
        return np.array([float(x) for x in euler_str.split()]) if euler_str else np.zeros(3)

    def _parse_quat(self, quat_str: str) -> np.ndarray:
        return np.array([float(x) for x in quat_str.split()]) if quat_str else np.array([1.0, 0.0, 0.0, 0.0])

    def _quat_to_rotmat(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])

    def _euler_to_rotmat(self, euler: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = euler
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _body_transform(self, body: ET.Element, parent_transform: np.ndarray = np.eye(4)) -> np.ndarray:
        pos = self._parse_pos(body.get('pos'))
        euler = self._parse_euler(body.get('euler'))
        quat = self._parse_quat(body.get('quat'))
        T = np.eye(4)
        if body.get('quat') is not None:
            T[:3, :3] = self._quat_to_rotmat(quat)
        else:
            T[:3, :3] = self._euler_to_rotmat(euler)
        T[:3, 3] = pos
        return parent_transform @ T

    def _axis_from_defaults(self, class_name: str) -> str | None:
        for d in self.defaults:
            if d.get('class') == class_name:
                joint_elem = d.find('joint')
                if joint_elem is not None and joint_elem.get('axis') is not None:
                    return joint_elem.get('axis')
        return None

    def _axis_to_letter(self, axis_str: str) -> str:
        vals = np.array([float(x) for x in axis_str.split()])
        return ['x', 'y', 'z'][np.argmax(np.abs(vals))]

    def _build_actuator_map(self) -> dict[str, str]:
        actuator_map = {}
        for act in self.root.findall('.//actuator/*'):
            jname = act.get('joint')
            if jname:
                actuator_map[jname] = act.tag
        return actuator_map

    def _find_arm_chain(self, actuator_map: dict[str, str]) -> list[tuple[str, ET.Element, ET.Element, np.ndarray]]:
        worldbody = self.root.find('.//worldbody')

        def walk(body_elem, parent_T=np.eye(4)):
            results = []
            T_body = self._body_transform(body_elem, parent_T)
            joint = body_elem.find('joint')
            if joint is not None:
                jname = joint.get('name')
                if jname and jname in actuator_map and actuator_map[jname] == 'position':
                    jpos = self._parse_pos(joint.get('pos'))
                    T_joint = T_body.copy()
                    T_joint[:3, 3] = T_body[:3, :3] @ jpos + T_body[:3, 3]
                    results.append((jname, joint, body_elem, T_joint))
            for child in body_elem.findall('body'):
                results.extend(walk(child, T_body))
            return results

        return walk(worldbody)

    def _compute_link_offsets(self, arm_chain: list) -> list[list[float]]:
        offsets = []
        for i in range(len(arm_chain)):
            if i + 1 < len(arm_chain):
                R_i = arm_chain[i][3][:3, :3]
                offset = R_i.T @ (arm_chain[i + 1][3][:3, 3] - arm_chain[i][3][:3, 3])
                offsets.append([round(v, 6) for v in offset.tolist()])
            else:
                offsets.append([0.0, 0.0, 0.0])
        return offsets

    def generate(self) -> dict:
        actuator_map = self._build_actuator_map()

        arm_joints = [j for j, t in actuator_map.items() if t == 'position']
        drive_joints = [j for j, t in actuator_map.items() if t == 'motor']

        arm_chain = self._find_arm_chain(actuator_map)

        rot_axes = []
        for _, joint_elem, _, _ in arm_chain:
            axis = joint_elem.get('axis')
            if axis is None:
                class_name = joint_elem.get('class')
                if class_name:
                    axis = self._axis_from_defaults(class_name)
            if axis is None:
                axis = "0 0 1"
            rot_axes.append(self._axis_to_letter(axis))

        link_offsets = self._compute_link_offsets(arm_chain)
        ee_body_name = arm_chain[-1][2].get('name', '') if arm_chain else ""

        config = {"model": str(self.xml_path.as_posix()), "dt": 0.02}

        joint_groups = {}
        if arm_joints:
            joint_groups["arm_joints"] = arm_joints
        if drive_joints:
            joint_groups["drive_joints"] = drive_joints
        if joint_groups:
            config["joint_groups"] = joint_groups

        plants = []
        if arm_joints:
            plants.append({
                "type": "ArmRobot", "name": "arm", "num_dof": len(arm_joints),
                "joint_group": "arm_joints", "ee_body_name": ee_body_name,
                "rot_axes": rot_axes, "joint_offsets": link_offsets,
            })
        if drive_joints:
            plants.append({
                "type": "HolonomicMobileRobot", "name": "base",
                "num_wheels": len(drive_joints), "radius_robots": 0.12,
                "gamma": -1.57079632679, "radius_wheels": 0.09,
            })
        if plants:
            config["plants"] = plants

        return config


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    xml_path = sys.argv[1]
    if not Path(xml_path).exists():
        print(f"Error: file not found: {xml_path}", file=sys.stderr)
        sys.exit(1)

    generator = RobotConfigGenerator(xml_path)
    config = generator.generate()
    yaml.dump(config, sys.stdout, default_flow_style=None, sort_keys=False)
