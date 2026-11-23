"""Environment-backed cost model for oracle MPC experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from stable_worldmodel.masks import (
    calculate_giou,
    calculate_iou,
    generate_ogbcube_cube_mask,
    generate_pusht_ball_mask,
    generate_pusht_t_mask,
    generate_reacher_masks,
    generate_tworoom_circle_mask,
)


class OracleMaskModel(torch.nn.Module):
    """Evaluate action sequences by stepping the real environment.

    This intentionally does not use pixels or a learned predictor.  Each
    candidate is evaluated from the current state, and its final state is
    converted into the same analytic/renderer mask used for the goal.
    """

    def __init__(
        self,
        env_name: str,
        action_block: int,
        textual_query: list[str],
        cost_function: str = 'mask',
        weight_cost_strategy: str = 'area',
        action_processor: Any = None,
    ) -> None:
        super().__init__()
        if cost_function not in ('mask', 'mask_iou', 'mask_mse'):
            raise ValueError(
                "Oracle MPC supports only 'mask', 'mask_iou', and "
                "'mask_mse' costs."
            )
        self.env_name = env_name.lower()
        self.action_block = action_block
        self.textual_query = textual_query
        self.cost_function = cost_function
        self.weight_cost_strategy = weight_cost_strategy
        self.action_processor = action_processor
        self.envs = None

    def set_env(self, envs: Any) -> None:
        self.envs = envs

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        if self.envs is None:
            raise RuntimeError('OracleMaskModel.set_env must be called first.')
        if 'goal_mask' not in info_dict:
            raise ValueError("Oracle MPC requires 'goal_mask'.")

        batch_size, num_candidates = action_candidates.shape[:2]
        goals = info_dict['goal_mask']
        costs = torch.empty(
            (batch_size, num_candidates),
            device=action_candidates.device,
            dtype=torch.float32,
        )

        for env_idx in range(batch_size):
            goal_masks = torch.as_tensor(
                self._first_value(goals[env_idx], target_ndim=3)
            ).detach().cpu()
            weights = self._mask_weights(goal_masks)
            env_indices = info_dict.get('_env_indices')
            if env_indices is None:
                global_env_idx = env_idx
            else:
                env_indices = np.asarray(env_indices)
                global_env_idx = int(
                    env_indices[env_idx].reshape(-1)[0]
                )
            env = self.envs.envs[global_env_idx].unwrapped
            base_state = self._get_sim_state(env)
            env_costs = []
            for candidate in action_candidates[env_idx]:
                final_state = self._simulate(
                    env, base_state, candidate.detach().cpu().numpy()
                )
                try:
                    predicted = self._state_masks(env, final_state)
                finally:
                    self._set_sim_state(env, base_state)
                value = 0.0
                for pred, goal, weight in zip(predicted, goal_masks, weights):
                    if self.cost_function == 'mask_mse':
                        value += float(
                            F.mse_loss(
                                pred.float(), goal.float()
                            ).item()
                        )
                    else:
                        overlap = (
                            calculate_iou(pred, goal)
                            if self.cost_function == 'mask_iou'
                            else calculate_giou(pred, goal)
                        )
                        value += (1.0 - overlap) * float(weight)
                env_costs.append(value)
            costs[env_idx] = torch.as_tensor(
                env_costs, device=action_candidates.device
            )
        return costs

    @staticmethod
    def _first_value(value: torch.Tensor, target_ndim: int) -> torch.Tensor:
        # Solver expansion produces (env, candidate, time, ...). Select the
        # first candidate/time while retaining the payload dimensions.
        while value.ndim > target_ndim:
            value = value[0]
        return value

    def _mask_weights(self, goal_masks: torch.Tensor) -> torch.Tensor:
        if self.weight_cost_strategy == 'area':
            areas = goal_masks.float().sum(dim=(1, 2))
            total = areas.sum()
            return areas / total if total > 0 else torch.ones_like(areas)
        return torch.ones(goal_masks.shape[0])

    def _simulate(
        self, env: Any, base_state: Any, candidate: np.ndarray
    ) -> Any:
        self._set_sim_state(env, base_state)
        try:
            action_dim = candidate.shape[-1] // self.action_block
            actions = candidate.reshape(-1, self.action_block, action_dim)
            for action_group in actions:
                for action in action_group:
                    env.step(self._environment_action(action))
            return self._current_state(env)
        finally:
            self._set_sim_state(env, base_state)

    def _environment_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_processor is None:
            return action
        return self.action_processor.inverse_transform(
            action.reshape(1, -1)
        )[0]

    def _get_sim_state(self, env: Any) -> Any:
        if 'cube' in self.env_name:
            return (
                np.asarray(env._data.qpos).copy(),
                np.asarray(env._data.qvel).copy(),
                np.asarray(env._prev_qpos).copy()
                if hasattr(env, '_prev_qpos')
                else None,
                np.asarray(env._prev_qvel).copy()
                if hasattr(env, '_prev_qvel')
                else None,
            )
        if 'reacher' in self.env_name:
            physics = env.env.physics
            return (
                np.asarray(physics.data.qpos).copy(),
                np.asarray(physics.data.qvel).copy(),
            )
        if hasattr(env, '_get_obs'):
            return np.asarray(env._get_obs()).copy()
        raise NotImplementedError(
            f'No state getter is available for {type(env).__name__}.'
        )

    def _set_sim_state(self, env: Any, state: Any) -> None:
        if 'cube' in self.env_name:
            qpos, qvel, prev_qpos, prev_qvel = state
            self._set_state(env, qpos, qvel)
            if prev_qpos is not None:
                env._prev_qpos = prev_qpos.copy()
            if prev_qvel is not None:
                env._prev_qvel = prev_qvel.copy()
        elif 'reacher' in self.env_name:
            qpos, qvel = state
            self._set_state(env, qpos, qvel)
        else:
            self._set_state(env, state)

    def _set_state(
        self,
        env: Any,
        state: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        state = np.asarray(state)
        if 'cube' in self.env_name:
            if velocity is None:
                velocity = np.zeros(env._model.nv)
            env.set_state(state, np.asarray(velocity))
        elif 'reacher' in self.env_name:
            physics = env.env.physics
            if velocity is None:
                velocity = np.zeros(physics.model.nv)
            env.set_state(state, np.asarray(velocity))
        elif 'tworoom' in self.env_name:
            env._set_state(state[:2])
        elif hasattr(env, '_set_state'):
            env._set_state(state)
        else:
            raise NotImplementedError(
                f'No state setter is available for {type(env).__name__}.'
            )

    def _current_state(self, env: Any) -> np.ndarray:
        if 'cube' in self.env_name:
            return np.asarray(env._data.qpos).copy()
        if 'reacher' in self.env_name:
            return np.asarray(env.env.physics.data.qpos).copy()
        if hasattr(env, '_get_obs'):
            return np.asarray(env._get_obs()).copy()
        raise NotImplementedError(
            f'No state getter is available for {type(env).__name__}.'
        )

    def _state_masks(self, env: Any, state: np.ndarray) -> list[torch.Tensor]:
        if 'pusht' in self.env_name:
            masks = []
            for query in self.textual_query:
                if 'blue' in query or 'circle' in query or 'ball' in query:
                    masks.append(generate_pusht_ball_mask(torch.as_tensor(state)))
                else:
                    masks.append(generate_pusht_t_mask(torch.as_tensor(state)))
            return masks
        if 'tworoom' in self.env_name:
            return [generate_tworoom_circle_mask(torch.as_tensor(state))]
        if 'reacher' in self.env_name:
            state_dict = {'goal_qpos': np.asarray([state])}
            circle, stick = generate_reacher_masks(state_dict, 0)
            return [
                stick if 'stick' in query else circle
                for query in self.textual_query
            ]
        if 'cube' in self.env_name:
            return [generate_ogbcube_cube_mask(env, state)]
        raise NotImplementedError(
            f'Oracle masks are not implemented for {self.env_name}.'
        )
