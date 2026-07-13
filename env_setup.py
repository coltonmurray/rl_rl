from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from rlgym.api import StateMutator
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym_ppo.util import RLGymV2GymWrapper

from rewards import NectoLiteReward


def _uniform_range(rng: np.random.Generator, values) -> float:
    low, high = values
    return float(rng.uniform(float(low), float(high)))


class KickoffOrChaseMutator(StateMutator[GameState]):
    """Mix standard kickoffs with simple chase/recovery starts."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.kickoff = KickoffMutator()
        self.rng = np.random.default_rng(config.get("seed"))

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        if self.rng.random() < float(self.cfg["kickoff_probability"]):
            self.kickoff.apply(state, shared_info)
            return

        self._apply_chase_state(state)

    def _apply_chase_state(self, state: GameState) -> None:
        ball_x = self.rng.uniform(-float(self.cfg["ball_x_abs_max"]), float(self.cfg["ball_x_abs_max"]))
        ball_y = self.rng.uniform(-float(self.cfg["ball_y_abs_max"]), float(self.cfg["ball_y_abs_max"]))
        ball_speed = self.rng.uniform(0.0, float(self.cfg["ball_speed_max"]))
        ball_angle = self.rng.uniform(0.0, 2.0 * np.pi)

        state.ball.position = np.asarray([ball_x, ball_y, common_values.BALL_RESTING_HEIGHT], dtype=np.float32)
        state.ball.linear_velocity = np.asarray(
            [np.cos(ball_angle) * ball_speed, np.sin(ball_angle) * ball_speed, 0.0],
            dtype=np.float32,
        )
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        for car in state.cars.values():
            self._place_car_for_chase(car, state.ball.position)

    def _place_car_for_chase(self, car, ball_pos: np.ndarray) -> None:
        team_sign = 1.0 if car.team_num == common_values.BLUE_TEAM else -1.0
        distance = _uniform_range(self.rng, self.cfg["car_distance_range"])
        lateral = _uniform_range(self.rng, self.cfg["car_lateral_range"])

        car_x = float(ball_pos[0] + lateral)
        car_y = float(ball_pos[1] - team_sign * distance)
        car_x = float(np.clip(car_x, -float(self.cfg["car_x_abs_max"]), float(self.cfg["car_x_abs_max"])))
        car_y = float(np.clip(car_y, -float(self.cfg["car_y_abs_max"]), float(self.cfg["car_y_abs_max"])))

        car.physics.position = np.asarray([car_x, car_y, 17.0], dtype=np.float32)

        to_ball = ball_pos[:2] - car.physics.position[:2]
        yaw = float(np.arctan2(to_ball[1], to_ball[0]))
        yaw += self.rng.uniform(-float(self.cfg["car_yaw_noise"]), float(self.cfg["car_yaw_noise"]))
        car.physics.euler_angles = np.asarray([0.0, yaw, 0.0], dtype=np.float32)

        speed = _uniform_range(self.rng, self.cfg["car_speed_range"])
        car.physics.linear_velocity = np.asarray([np.cos(yaw) * speed, np.sin(yaw) * speed, 0.0], dtype=np.float32)
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.boost_amount = _uniform_range(self.rng, self.cfg["boost_range"])
        car.on_ground = True
        car.ball_touches = 0
        car.bump_victim_id = None
        car.demo_respawn_timer = 0.0
        car.supersonic_time = 0.0
        car.boost_active_time = 0.0
        car.handbrake = 0.0

        car.has_jumped = False
        car.is_holding_jump = False
        car.is_jumping = False
        car.jump_time = 0.0
        car.has_flipped = False
        car.has_double_jumped = False
        car.air_time_since_jump = 0.0
        car.flip_time = 0.0
        car.flip_torque = np.zeros(3, dtype=np.float32)
        car.is_autoflipping = False
        car.autoflip_timer = 0.0
        car.autoflip_direction = 0.0


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_obs_builder(config: Dict[str, Any]) -> DefaultObs:
    team_size = int(config["env"]["team_size"])
    return DefaultObs(
        zero_padding=team_size,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )


def make_rlgym_env(config: Dict[str, Any]):
    env_cfg = config["env"]
    team_size = int(env_cfg["team_size"])
    orange_size = team_size if env_cfg["spawn_opponents"] else 0
    action_repeat = int(env_cfg["action_repeat"])

    action_parser = RepeatAction(
        LookupTableAction(),
        repeats=action_repeat,
    )
    truncation = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=env_cfg["no_touch_timeout_seconds"]),
        TimeoutCondition(timeout_seconds=env_cfg["game_timeout_seconds"]),
    )
    reset_mutator = KickoffMutator()
    curriculum_cfg = config.get("state_curriculum", {})
    if curriculum_cfg.get("enabled", False):
        reset_mutator = KickoffOrChaseMutator(curriculum_cfg)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=team_size, orange_size=orange_size),
        reset_mutator,
    )

    from rlgym.api import RLGym

    return RLGym(
        state_mutator=state_mutator,
        obs_builder=make_obs_builder(config),
        action_parser=action_parser,
        reward_fn=NectoLiteReward(config["necto_lite_reward"]),
        termination_cond=GoalCondition(),
        truncation_cond=truncation,
        transition_engine=RocketSimEngine(),
    )


def make_gym_env(config: Dict[str, Any]) -> RLGymV2GymWrapper:
    return RLGymV2GymWrapper(make_rlgym_env(config))


def action_count() -> int:
    return len(LookupTableAction.make_lookup_table())
