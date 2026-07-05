from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym_ppo.util import RLGymV2GymWrapper


OBS_NAMES = [
    "car_pos_x",
    "car_pos_y",
    "car_pos_z",
    "car_vel_x",
    "car_vel_y",
    "car_vel_z",
    "car_rot_pitch",
    "car_rot_yaw",
    "car_rot_roll",
    "car_ang_vel_x",
    "car_ang_vel_y",
    "car_ang_vel_z",
    "car_boost",
    "car_on_ground",
    "ball_pos_x",
    "ball_pos_y",
    "ball_pos_z",
    "ball_vel_x",
    "ball_vel_y",
    "ball_vel_z",
    "rel_ball_x",
    "rel_ball_y",
    "rel_ball_z",
    "rel_ball_vel_x",
    "rel_ball_vel_y",
    "rel_ball_vel_z",
    "opp_pos_x",
    "opp_pos_y",
    "opp_pos_z",
    "opp_vel_x",
    "opp_vel_y",
    "opp_vel_z",
    "opp_boost",
    "opp_on_ground",
]


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class StructuredObs(ObsBuilder[AgentID, np.ndarray, GameState, Tuple[str, int]]):
    """Small fixed observation for early model-based experiments."""

    def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
        return "real", len(OBS_NAMES)

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def build_obs(
        self,
        agents: List[AgentID],
        state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, np.ndarray]:
        return {agent: self._build_one(agent, state) for agent in agents}

    def _build_one(self, agent: AgentID, state: GameState) -> np.ndarray:
        car = state.cars[agent]
        inverted = car.team_num == common_values.ORANGE_TEAM
        car_physics = car.inverted_physics if inverted else car.physics
        ball = state.inverted_ball if inverted else state.ball

        enemy = self._first_enemy(agent, state)
        if enemy is None:
            enemy_features = np.zeros(8, dtype=np.float32)
        else:
            enemy_physics = enemy.inverted_physics if inverted else enemy.physics
            enemy_features = np.concatenate(
                [
                    self._norm_pos(enemy_physics.position),
                    self._norm_vel(enemy_physics.linear_velocity),
                    np.asarray(
                        [enemy.boost_amount / 100.0, float(enemy.on_ground)],
                        dtype=np.float32,
                    ),
                ]
            )

        obs = np.concatenate(
            [
                self._norm_pos(car_physics.position),
                self._norm_vel(car_physics.linear_velocity),
                self._norm_rot(self._safe_euler(car_physics)),
                self._norm_ang_vel(car_physics.angular_velocity),
                np.asarray([car.boost_amount / 100.0, float(car.on_ground)]),
                self._norm_pos(ball.position),
                self._norm_vel(ball.linear_velocity),
                self._norm_pos(ball.position - car_physics.position),
                self._norm_vel(ball.linear_velocity - car_physics.linear_velocity),
                enemy_features,
            ]
        )
        return obs.astype(np.float32)

    def _first_enemy(self, agent: AgentID, state: GameState) -> Car | None:
        team = state.cars[agent].team_num
        for other_id, other_car in state.cars.items():
            if other_id != agent and other_car.team_num != team:
                return other_car
        return None

    def _safe_euler(self, physics: Any) -> np.ndarray:
        try:
            return physics.euler_angles
        except ValueError:
            return np.zeros(3, dtype=np.float32)

    def _norm_pos(self, value: np.ndarray) -> np.ndarray:
        return value * np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ],
            dtype=np.float32,
        )

    def _norm_vel(self, value: np.ndarray) -> np.ndarray:
        return value * (1 / common_values.CAR_MAX_SPEED)

    def _norm_ang_vel(self, value: np.ndarray) -> np.ndarray:
        return value * (1 / common_values.CAR_MAX_ANG_VEL)

    def _norm_rot(self, value: np.ndarray) -> np.ndarray:
        return value * (1 / np.pi)


def make_rlgym_env(config: Dict[str, Any]):
    env_cfg = config["env"]
    reward_cfg = config["reward_weights"]
    team_size = int(env_cfg["team_size"])
    orange_size = team_size if env_cfg["spawn_opponents"] else 0

    action_parser = RepeatAction(
        LookupTableAction(),
        repeats=int(env_cfg["action_repeat"]),
    )
    reward_fn = CombinedReward(
        (GoalReward(), float(reward_cfg["goal"])),
        (TouchReward(), float(reward_cfg["touch"])),
    )
    truncation = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=env_cfg["no_touch_timeout_seconds"]),
        TimeoutCondition(timeout_seconds=env_cfg["game_timeout_seconds"]),
    )
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=team_size, orange_size=orange_size),
        KickoffMutator(),
    )

    from rlgym.api import RLGym

    return RLGym(
        state_mutator=state_mutator,
        obs_builder=StructuredObs(),
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=GoalCondition(),
        truncation_cond=truncation,
        transition_engine=RocketSimEngine(),
    )


def make_gym_env(config: Dict[str, Any]) -> RLGymV2GymWrapper:
    return RLGymV2GymWrapper(make_rlgym_env(config))


def action_count() -> int:
    return len(LookupTableAction.make_lookup_table())
