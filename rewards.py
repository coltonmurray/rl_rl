from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState


def _safe_norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = _safe_norm(a) * _safe_norm(b)
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


class NectoLiteReward(RewardFunction[AgentID, GameState, float]):
    """Dense model-free reward inspired by Necto's reward shaping.

    This is intentionally much smaller than Necto's original reward stack. It keeps
    the signals that fit our current RLGym v2 prototype without bringing in
    rocket-learn, Redis workers, or scoreboard-specific machinery.
    """

    def __init__(self, config: Dict[str, float]):
        self.cfg = config
        self.last_ball_pos = None
        self.last_ball_vel = None
        self.last_boost = {}
        self.last_demoed = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_ball_pos = initial_state.ball.position.copy()
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_boost = {agent: initial_state.cars[agent].boost_amount for agent in agents}
        self.last_demoed = {agent: initial_state.cars[agent].is_demoed for agent in agents}

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        raw = {agent: self._agent_reward(agent, state, is_truncated) for agent in agents}
        rewards = self._blend_team_rewards(raw, agents, state)

        self.last_ball_pos = state.ball.position.copy()
        self.last_ball_vel = state.ball.linear_velocity.copy()
        self.last_boost = {agent: state.cars[agent].boost_amount for agent in agents}
        self.last_demoed = {agent: state.cars[agent].is_demoed for agent in agents}
        return rewards

    def _agent_reward(self, agent: AgentID, state: GameState, is_truncated: Dict[AgentID, bool]) -> float:
        car = state.cars[agent]
        team_sign = 1.0 if car.team_num == common_values.BLUE_TEAM else -1.0
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        car_pos = car.physics.position

        reward = float(self.cfg.get("step_penalty", 0.0))

        if state.goal_scored:
            reward += self.cfg["goal"] if state.scoring_team == car.team_num else -self.cfg["concede"]
        elif is_truncated.get(agent, False):
            reward -= self.cfg.get("no_touch_timeout_penalty", 0.0)

        if self.last_ball_pos is not None:
            prev_quality = team_sign * self.last_ball_pos[1] / common_values.BACK_NET_Y
            next_quality = team_sign * ball_pos[1] / common_values.BACK_NET_Y
            reward += self.cfg["ball_progress"] * (next_quality - prev_quality)

        car_to_ball = ball_pos - car_pos
        dist = _safe_norm(car_to_ball)
        reward += self.cfg["car_ball_dist"] * np.exp(-dist / 1410.0)

        target_goal = (
            np.asarray(common_values.ORANGE_GOAL_BACK, dtype=np.float32)
            if car.team_num == common_values.BLUE_TEAM
            else np.asarray(common_values.BLUE_GOAL_BACK, dtype=np.float32)
        )
        reward += self.cfg["alignment"] * _cosine(car_to_ball, target_goal - car_pos)

        if car.ball_touches > 0:
            reward += self.cfg["touch"]
            if self.last_ball_vel is not None:
                reward += self.cfg["touch_accel"] * _safe_norm(ball_vel - self.last_ball_vel) / common_values.CAR_MAX_SPEED
            height = np.clip((ball_pos[2] - common_values.BALL_RADIUS) / common_values.CEILING_Z, 0.0, 1.0)
            reward += self.cfg["touch_height"] * height

        last_boost = self.last_boost.get(agent, car.boost_amount)
        boost_delta = np.sqrt(max(car.boost_amount, 0.0) / 100.0) - np.sqrt(max(last_boost, 0.0) / 100.0)
        if boost_delta >= 0:
            reward += self.cfg["boost_gain"] * boost_delta
        else:
            reward += self.cfg["boost_spend"] * boost_delta

        if car.is_demoed and not self.last_demoed.get(agent, False):
            reward -= self.cfg["demoed"]

        return float(reward)

    def _blend_team_rewards(self, rewards: Dict[AgentID, float], agents: List[AgentID], state: GameState) -> Dict[AgentID, float]:
        team_spirit = float(self.cfg.get("team_spirit", 0.0))
        opponent_punish = float(self.cfg.get("opponent_punish", 0.0))
        if team_spirit == 0.0 and opponent_punish == 0.0:
            return rewards

        by_team = {
            common_values.BLUE_TEAM: [agent for agent in agents if state.cars[agent].team_num == common_values.BLUE_TEAM],
            common_values.ORANGE_TEAM: [agent for agent in agents if state.cars[agent].team_num == common_values.ORANGE_TEAM],
        }
        means = {
            team: float(np.mean([rewards[agent] for agent in team_agents])) if team_agents else 0.0
            for team, team_agents in by_team.items()
        }

        blended = {}
        for agent in agents:
            team = state.cars[agent].team_num
            opponent = common_values.ORANGE_TEAM if team == common_values.BLUE_TEAM else common_values.BLUE_TEAM
            own = rewards[agent]
            blended[agent] = (
                (1.0 - team_spirit) * own
                + team_spirit * means[team]
                - opponent_punish * means[opponent]
            )
        return blended
