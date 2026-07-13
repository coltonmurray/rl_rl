from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import FieldInfoPacket, GameTickPacket


ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "outputs" / "rlbot" / "policy.npz"

BLUE_TEAM = 0
ORANGE_TEAM = 1
SIDE_WALL_X = 4096.0
BACK_NET_Y = 6000.0
CEILING_Z = 2044.0
CAR_MAX_SPEED = 2300.0
CAR_MAX_ANG_VEL = 5.5
DEMO_RESPAWN_SECONDS = 3.0
INV_VEC = np.asarray([-1.0, -1.0, 1.0], dtype=np.float32)
POS_COEF = np.asarray([1 / SIDE_WALL_X, 1 / BACK_NET_Y, 1 / CEILING_Z], dtype=np.float32)
LIN_VEL_COEF = 1 / CAR_MAX_SPEED
ANG_VEL_COEF = 1 / CAR_MAX_ANG_VEL
BOOST_COEF = 1 / 100.0
PAD_TIMER_COEF = 1 / 10.0

BOOST_LOCATIONS = (
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (-940.0, 3308.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
)


def vec3(value) -> np.ndarray:
    return np.asarray([value.x, value.y, value.z], dtype=np.float32)


def angular_velocity(physics) -> np.ndarray:
    if hasattr(physics, "angular_velocity"):
        return vec3(physics.angular_velocity)
    return np.zeros(3, dtype=np.float32)


def maybe_invert(value: np.ndarray, inverted: bool) -> np.ndarray:
    return value * INV_VEC if inverted else value


def orientation_vectors(rotation):
    pitch = float(rotation.pitch)
    yaw = float(rotation.yaw)
    roll = float(rotation.roll)

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    forward = np.asarray([cp * cy, cp * sy, sp], dtype=np.float32)
    up = np.asarray([-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr], dtype=np.float32)
    return forward, up


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class NumpyPolicy:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing exported policy {path}. Run: python export_rlbot_policy.py --config config.yaml"
            )
        data = np.load(str(path), allow_pickle=False)
        self.layer_count = int(data["layer_count"])
        self.weights = [data[f"w{i}"].astype(np.float32) for i in range(self.layer_count)]
        self.biases = [data[f"b{i}"].astype(np.float32) for i in range(self.layer_count)]
        self.obs_mean = data["obs_mean"].astype(np.float32)
        self.obs_std = data["obs_std"].astype(np.float32)
        self.lookup_table = data["lookup_table"].astype(np.float32)
        self.tick_skip = int(data["tick_skip"])
        self.deterministic = bool(data["deterministic"])
        self.checkpoint = str(data["checkpoint"])
        self.checkpoint_timesteps = int(data["checkpoint_timesteps"])

    def action(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs.astype(np.float32) - self.obs_mean) / self.obs_std, -5.0, 5.0)
        for i in range(self.layer_count):
            x = self.weights[i].dot(x) + self.biases[i]
            if i < self.layer_count - 1:
                x = relu(x)
        if self.deterministic:
            action_index = int(np.argmax(x))
        else:
            action_index = int(np.random.choice(len(x), p=softmax(x)))
        return self.lookup_table[action_index]


class RLRLPPOBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.policy = None
        self.tick_skip = 8
        self.boost_order: List[int] = []
        self.controls = SimpleControllerState()
        self.current_action = np.zeros(8, dtype=np.float32)
        self.ticks = self.tick_skip
        self.prev_time = 0.0

    def initialize_agent(self):
        self.policy = NumpyPolicy(POLICY_PATH)
        self.tick_skip = self.policy.tick_skip
        self.boost_order = self._build_boost_order(self.get_field_info())
        self.controls = SimpleControllerState()
        self.current_action = np.zeros(8, dtype=np.float32)
        self.ticks = self.tick_skip
        self.prev_time = 0.0
        print(f"RL_RL loaded exported policy: {POLICY_PATH}")
        print(f"RL_RL checkpoint: {self.policy.checkpoint}")
        print(f"RL_RL checkpoint timesteps: {self.policy.checkpoint_timesteps}")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if self.policy is None:
            return self.controls

        cur_time = packet.game_info.seconds_elapsed
        ticks_elapsed = round((cur_time - self.prev_time) * 120) if self.prev_time else self.tick_skip
        self.prev_time = cur_time
        self.ticks += ticks_elapsed

        if self.ticks >= self.tick_skip and not packet.game_info.is_match_ended:
            self.ticks = 0
            obs = self._build_obs(packet)
            self.current_action = self.policy.action(obs).astype(np.float32)
            self._update_controls(self.current_action)

        return self.controls

    def _build_boost_order(self, field_info: FieldInfoPacket) -> List[int]:
        packet_locations = np.asarray(
            [vec3(field_info.boost_pads[i].location) for i in range(field_info.num_boosts)],
            dtype=np.float32,
        )
        order = []
        for loc in np.asarray(BOOST_LOCATIONS, dtype=np.float32):
            distances = np.linalg.norm(packet_locations - loc, axis=1)
            order.append(int(np.argmin(distances)))
        return order

    def _boost_timers(self, packet: GameTickPacket, inverted: bool) -> np.ndarray:
        timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float32)
        for rlgym_idx, packet_idx in enumerate(self.boost_order):
            if packet_idx < packet.num_boost:
                pad = packet.game_boosts[packet_idx]
                timers[rlgym_idx] = 0.0 if pad.is_active else float(pad.timer)
        return timers[::-1] if inverted else timers

    def _build_obs(self, packet: GameTickPacket) -> np.ndarray:
        car = packet.game_cars[self.index]
        inverted = car.team == ORANGE_TEAM
        ball = packet.game_ball.physics

        ball_pos = maybe_invert(vec3(ball.location), inverted)
        ball_vel = maybe_invert(vec3(ball.velocity), inverted)
        ball_ang = maybe_invert(angular_velocity(ball), inverted)

        obs_parts = [
            ball_pos * POS_COEF,
            ball_vel * LIN_VEL_COEF,
            ball_ang * ANG_VEL_COEF,
            self._boost_timers(packet, inverted) * PAD_TIMER_COEF,
            self._partial_obs(car),
            self._car_obs(car, inverted),
        ]

        opponents = [
            packet.game_cars[i]
            for i in range(packet.num_cars)
            if i != self.index and packet.game_cars[i].team != car.team
        ]
        if opponents:
            obs_parts.append(self._car_obs(opponents[0], inverted))
        else:
            obs_parts.append(np.zeros(20, dtype=np.float32))

        return np.concatenate(obs_parts).astype(np.float32)

    def _partial_obs(self, car) -> np.ndarray:
        on_ground = bool(car.has_wheel_contact)
        has_jumped = bool(car.jumped)
        has_double_jumped = bool(car.double_jumped)
        can_flip = (not on_ground) and has_jumped and not has_double_jumped
        return np.asarray(
            [
                0.0,
                float(self.current_action[7] > 0),
                float(has_jumped),
                0.0,
                float(has_double_jumped),
                0.0,
                float(has_double_jumped),
                float(can_flip),
                0.0,
            ],
            dtype=np.float32,
        )

    def _car_obs(self, car, inverted: bool) -> np.ndarray:
        physics = car.physics
        pos = maybe_invert(vec3(physics.location), inverted)
        lin_vel = maybe_invert(vec3(physics.velocity), inverted)
        ang_vel = maybe_invert(angular_velocity(physics), inverted)
        forward, up = orientation_vectors(physics.rotation)
        forward = maybe_invert(forward, inverted)
        up = maybe_invert(up, inverted)
        demo_respawn = DEMO_RESPAWN_SECONDS if car.is_demolished else 0.0
        is_boosting = float(self.current_action[6] > 0)
        return np.concatenate(
            [
                pos * POS_COEF,
                forward,
                up,
                lin_vel * LIN_VEL_COEF,
                ang_vel * ANG_VEL_COEF,
                np.asarray(
                    [
                        float(car.boost) * BOOST_COEF,
                        demo_respawn,
                        float(car.has_wheel_contact),
                        is_boosting,
                        float(car.is_super_sonic),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def _update_controls(self, action: np.ndarray) -> None:
        self.controls.throttle = float(action[0])
        self.controls.steer = float(action[1])
        self.controls.pitch = float(action[2])
        self.controls.yaw = float(action[3])
        self.controls.roll = float(action[4])
        self.controls.jump = bool(action[5] > 0)
        self.controls.boost = bool(action[6] > 0)
        self.controls.handbrake = bool(action[7] > 0)
