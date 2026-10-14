#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.types import RobotObservation

from .utils import _LazyAsyncVectorEnv, parse_camera_names


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


# LIBERO-plus perturbation variants encode the perturbation in the filename
# but on disk only the base `.pruned_init` exists — strip the suffix to match
# LIBERO-plus's own suite.get_task_init_states() (we reimplement it here so we
# can pass weights_only=False for PyTorch 2.6+ numpy pickles).
_LIBERO_PERTURBATION_SUFFIX_RE = re.compile(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+")


def get_task_init_states(task_suite: Any, i: int, is_libero_plus: bool = False) -> np.ndarray:
    task = task_suite.tasks[i]
    filename = Path(task.init_states_file)
    root = Path(get_libero_path("init_states"))

    if not is_libero_plus:
        init_states_path = root / task.problem_folder / filename.name
        return torch.load(init_states_path, weights_only=False)  # nosec B614

    # LIBERO-plus: `_add_` / `_level` variants store extra-object layouts under
    # libero_newobj/ as a flat array that must be reshaped to (1, -1).
    if "_add_" in filename.name or "_level" in filename.name:
        init_states_path = root / "libero_newobj" / task.problem_folder / filename.name
        init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
        return init_states.reshape(1, -1)

    # LIBERO-plus perturbation variants encode the perturbation in the filename
    # but on disk only the base `.pruned_init` exists — strip the suffix to match.
    stripped = _LIBERO_PERTURBATION_SUFFIX_RE.sub("", filename.stem) + filename.suffix
    init_states_path = root / task.problem_folder / stripped
    return torch.load(init_states_path, weights_only=False)  # nosec B614


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_freq: int = 20,
        control_mode: str = "relative",
        is_libero_plus: bool = False,
        gripper_mode: str = "threshold",
        gripper_eps: float = 2e-3,
        gripper_lead: int = 0,
        gripper_close_level: float = 0.032,
        gripper_open_level: float = 0.008,
        joint_protocol: str = "v0",
    ):
        super().__init__()
        if control_freq <= 0:
            raise ValueError(f"control_freq must be positive, got {control_freq}")
        self.task_id = task_id
        self.is_libero_plus = is_libero_plus
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.control_freq = control_freq
        # 关节模式下的夹爪反解（见 docs/JOINT_CONTROL_PROTOCOL.md）
        if gripper_mode not in ("threshold", "velocity", "hysteresis"):
            raise ValueError(f"unknown gripper_mode {gripper_mode!r}")
        # 闭环关节协议（D24）：v0 = 纯 PD（T01/T02 事实协议，逐比特可复现）；
        # v1 = 参考前馈。实现见 lerobot/envs/joint_protocol.py。
        if joint_protocol not in ("v0", "v1"):
            raise ValueError(f"unknown joint_protocol {joint_protocol!r}; expected 'v0' or 'v1'")
        self.joint_protocol = joint_protocol
        self._joint_protocol_dt = 1.0 / float(control_freq)
        self.gripper_mode = gripper_mode
        self.gripper_eps = gripper_eps
        self.gripper_lead = gripper_lead
        self.gripper_close_level = gripper_close_level
        self.gripper_open_level = gripper_open_level
        self._gripper_prev: float | None = None
        self._gripper_cmd: float = 1.0
        self._gripper_hist: list[float] = []
        self.episode_index = episode_index
        self.episode_length = episode_length
        # Load once and keep
        self._init_states = (
            get_task_init_states(task_suite, self.task_id, is_libero_plus=self.is_libero_plus)
            if self.init_states
            else None
        )
        self._reset_stride = n_envs  # when performing a reset, append `_reset_stride` to `init_state_id`.

        self.init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        # Extract task metadata without allocating GPU resources (safe before fork).
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        self._task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self._env: OffScreenRenderEnv | None = (
            None  # deferred — created on first reset() inside the worker subprocess
        )

        default_steps = 500
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            if self.episode_length is None
            else self.episode_length
        )
        self.control_mode = control_mode
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    "quat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                }
                            ),
                        }
                    ),
                }
            )

        # joint 模式动作是 8D(7 绝对关节角 + 1 gripper), 其余(EEF delta)是 7D。
        # EEF 动作归一化在 [-1,1]; 绝对关节角可超出该范围(如 -2.5rad), 故 joint 模式放宽边界,
        # 避免任何按 action_space 的裁剪破坏绝对角命令。
        if self.control_mode == "joint":
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
            )

    def _ensure_env(self) -> None:
        """Create the underlying OffScreenRenderEnv on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own clean EGL context rather than inheriting a stale one from the
        parent process (which causes EGL_BAD_CONTEXT crashes with AsyncVectorEnv).
        """
        if self._env is not None:
            return
        env_kwargs = dict(
            bddl_file_name=self._task_bddl_file,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            control_freq=self.control_freq,
        )
        # "joint" 模式: 用 robosuite 的 JOINT_POSITION 控制器(control_dim=7 关节 + 1 gripper = 8D),
        # 供输出关节角的 policy(如关节空间微调的 smolvla)使用。EEF 模式默认 OSC_POSE。
        if self.control_mode == "joint":
            env_kwargs["controller"] = "JOINT_POSITION"
        env = OffScreenRenderEnv(**env_kwargs)
        env.reset()
        self._env = env
        if self.control_mode == "joint":
            self._patch_joint_controllers()

    def _patch_joint_controllers(self) -> None:
        """给 JOINT_POSITION 控制器打关节协议补丁（v0 = 绝对目标；v1 = 绝对目标 + 参考前馈）。

        关节空间数据集的 action = 绝对 joint_state(见 preprocess_libero_joint.py),
        而 robosuite JointPositionController.set_goal 默认把输入按 output_max 缩放成增量。
        协议实现集中在 `lerobot/envs/joint_protocol.py`（D24：全组唯一实现）。
        """
        assert self._env is not None
        from lerobot.envs.joint_protocol import attach, reset_protocol_state

        dt = 1.0 / float(self.control_freq)
        for robot in self._env.robots:
            attach(robot, self.joint_protocol, dt)
            reset_protocol_state(robot)

    def render(self):
        self._ensure_env()
        raw_obs = self._env.env._get_observations()
        pixels = self._format_raw_obs(raw_obs)["pixels"]
        image = next(iter(pixels.values()))
        image = image[::-1, ::-1]  # flip both H and W for visualization
        return image

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        assert self._env is not None, "_format_raw_obs called before _ensure_env()"
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            images[self.camera_name_mapping[camera_name]] = image

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        # rotation matrix from controller
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")
        obs = {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,  # (3,)
                    "quat": eef_quat,  # (4,)
                    "mat": eef_mat,  # (3, 3)
                },
                "gripper": {
                    "qpos": gripper_qpos,  # (2,)
                    "qvel": gripper_qvel,  # (2,)
                },
                "joints": {
                    "pos": joint_pos,  # (7,)
                    "vel": joint_vel,  # (7,)
                },
            },
        }
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    f"gripper_qpos={gripper_qpos is not None}"
                )
            return obs

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        super().reset(seed=seed)
        self.reset_gripper_decoder()
        self._env.seed(seed)
        raw_obs = self._env.reset()
        # LIBERO 用 hard_reset=True: 每次 self._env.reset() 都【重建】机器人+控制器对象,
        # 会把 _ensure_env 里打的 set_goal monkey-patch 丢掉(实测 reset 后 controller 是新对象、
        # 未 patch)。必须在这里对新控制器重新打补丁, 否则 joint 模式退回默认的 ±0.05rad 增量
        # 缩放, 绝对关节角命令失效(表现为跟踪误差随步数发散)。
        if self.control_mode == "joint":
            self._patch_joint_controllers()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
            self.init_state_id += self._reset_stride  # Change init_state_id when reset

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for _ in range(self.num_steps_wait):
            if self.control_mode == "joint":
                # 关节模式 no-op = 保持当前关节角(8D: 7 绝对角 + gripper 合)。
                # get_libero_dummy_action() 是 7D EEF 格式, 维度对不上 JOINT_POSITION(8D)。
                cur_q = np.asarray(self._env.robots[0]._joint_positions, dtype=np.float32)
                noop = np.concatenate([cur_q, [1.0]]).astype(np.float32)
                raw_obs, _, _, _ = self._env.step(noop)
            else:
                raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        elif self.control_mode == "joint":
            # JOINT_POSITION 控制器无 use_delta 属性; step() 里直接注入绝对 qpos, 这里无需设置。
            pass
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def _gripper_action(self, value: float) -> float:
        """把"下一帧手指 qpos"反解成 robosuite 夹爪命令（-1 开 / +1 合）。"""
        if self.gripper_mode == "threshold":
            self._gripper_prev = value
            self._gripper_cmd = -1.0 if value > 0.02 else 1.0
            return self._gripper_cmd
        if self.gripper_mode == "hysteresis":
            # 迟滞阈值：`q>0.02` 的幅值判据（抗策略噪声）+ 提前 ~3 帧切命令
            # （手指过渡段约 0.004/帧，所以把"合"的门槛抬到 0.032）。
            self._gripper_prev = value
            if value < self.gripper_close_level:
                self._gripper_cmd = 1.0
            elif value > self.gripper_open_level:
                self._gripper_cmd = -1.0
            return self._gripper_cmd
        # velocity: 手指一动就说明命令刚切过（robosuite 是把 ±1 逐步积分到执行器目标的）
        if self._gripper_prev is None:
            self._gripper_cmd = -1.0 if value > 0.02 else 1.0
        else:
            delta = value - self._gripper_prev
            if delta > self.gripper_eps:
                self._gripper_cmd = -1.0
            elif delta < -self.gripper_eps:
                self._gripper_cmd = 1.0
        self._gripper_prev = value
        return self._gripper_cmd

    def reset_gripper_decoder(self) -> None:
        """新一轮 episode 前清掉夹爪解码器的状态。"""
        self._gripper_prev = None
        self._gripper_cmd = 1.0

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        if self.control_mode == "joint":
            # 关节模式 action = [7 绝对关节角, 1 gripper]。第 8 维是数据集里的**下一帧手指 qpos**
            # (≈0.04 张开 / ≈0 闭合)，要反解成 robosuite 的夹爪命令 (-1=开 / +1=合)。
            # 前 7 维绝对角由 patch 过的 set_goal 直接注入。
            # ★ base/T02：阈值反解会把命令系统性地推迟 ~5 帧（手指要走完 0.04→0.02 才被判到），
            #   与 PD 的跟踪滞后互相抵消，制造出"控制器越准越差"的假象。见 JOINT_CONTROL_PROTOCOL.md §3。
            action = action.copy()
            action[7] = self._gripper_action(float(action[7]))
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            self.reset()
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            camera_name_mapping=camera_name_mapping,
            is_libero_plus=is_libero_plus,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks

    camera_names = parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    is_async = env_cls is gym.vector.AsyncVectorEnv

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)
        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        # All tasks in a suite share identical observation/action spaces.
        # Probe once and reuse to avoid creating a temp env per task.
        cached_obs_space: spaces.Space | None = None
        cached_act_space: spaces.Space | None = None
        cached_metadata: dict[str, Any] | None = None

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
                camera_name_mapping=camera_name_mapping,
                is_libero_plus=is_libero_plus,
            )
            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                out[suite_name][tid] = lazy
            else:
                out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    return {suite: dict(task_map) for suite, task_map in out.items()}
