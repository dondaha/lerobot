"""闭环关节控制协议 v0 / v1 的**唯一实现**（D24）。

- **v0** = robosuite 原生 `JointPositionController`（PD, kp=50, ζ=1）+ 绝对目标注入。
  这是 T01/T02 全部闭环数字的事实协议，必须能逐比特复现。
- **v1** = v0 + **参考轨迹前馈**：kp=50、ζ=1、`v_clip=4 rad/s`、`a_clip=200 rad/s²`，
  参考从"上一条命令"斜坡到"当前命令"，力矩 `τ = M(q)·(kp·(ref−q) + kd·(v_ref−q̇) + a_ref) + bias`。

数学与离线回放实验台 `safe_vla/scripts/diagnostics/joint_ctrl_lab.py::_attach_lab_controller`
**逐行一致**（那边的 `ff="velacc"` 预设就是 v1）。闭环夹爪解码器由 `LiberoEnv.gripper_mode`
单独控制（v1 冻结组合见 `docs/JOINT_CONTROL_PROTOCOL.md`）。

实现方式是**实例级 monkey-patch**（不子类化）：LIBERO 用 `hard_reset=True`，每次 `env.reset()`
都会重建 robot/controller 对象，所以每到一次 reset 都要重新 attach 并清空每回合状态
（`v_ref`/`a_ref`/`prev_goal`/`ref_start`/`t_step`——不清会吃到上回合的 residue，第一步力矩尖峰）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

JOINT_PROTOCOLS = ("v0", "v1")

# v1 冻结参数（docs/JOINT_CONTROL_PROTOCOL.md §1/§5）
KP = 50.0
DAMPING_RATIO = 1.0
V_CLIP = 4.0
A_CLIP = 200.0


def _control_dim(ctrl) -> int:
    return int(getattr(ctrl, "control_dim", 7))


def patch_absolute_set_goal(robot) -> None:
    """v0：把动作当成【绝对】关节角（而不是 robosuite 默认的 ±output_max 增量）。

    数据集里 `action[t] = state[t+1]`（绝对关节角），而 `JointPositionController.set_goal`
    默认按 `output_max` 缩放成增量。这里用 `set_qpos=` 直接设绝对目标。
    """
    ctrl = robot.controller

    def abs_set_goal(action, set_qpos=None):  # noqa: ARG001 - 保持与 robosuite 同签名
        return type(ctrl).set_goal(ctrl, action, set_qpos=np.asarray(action, dtype=float))

    ctrl.set_goal = abs_set_goal
    ctrl._joint_protocol = "v0"


def attach_reference_feedforward(
    robot,
    dt: float,
    kp: float = KP,
    damping_ratio: float = DAMPING_RATIO,
    v_clip: float = V_CLIP,
    a_clip: float = A_CLIP,
) -> None:
    """v1：绝对目标 + 因果参考前馈（只需上一条命令，不预读未来）。

    `dt` = 控制周期（LIBERO 关节模式 20 Hz ⇒ 0.05 s）。
    """
    from robosuite.controllers.base_controller import Controller as _BaseController

    ctrl = robot.controller
    dim = _control_dim(ctrl)

    ctrl.kp = np.full(dim, float(kp))
    ctrl.kd = 2.0 * np.sqrt(ctrl.kp) * float(damping_ratio)
    ctrl.interpolator = None  # v1 用显式 ramp，不用 robosuite 的插值器

    st: dict[str, Any] = {
        "goal": None,
        "prev_goal": None,
        "v_ref": np.zeros(dim),
        "a_ref": np.zeros(dim),
        "t_step": 0.0,
        "ref_start": None,
        "n_goals": 0,
    }

    def abs_set_goal(action, set_qpos=None):  # noqa: ARG001
        q = np.asarray(action, dtype=float).ravel()[:dim]
        type(ctrl).set_goal(ctrl, q, set_qpos=q)
        st["n_goals"] += 1
        st["t_step"] = float(ctrl.sim.data.time)
        if st["goal"] is None:
            v_new = np.zeros_like(q)
            a_new = np.zeros_like(q)
            st["prev_goal"] = q.copy()
        else:
            v_new = np.clip((q - st["goal"]) / dt, -v_clip, v_clip)
            a_new = np.clip((v_new - st["v_ref"]) / dt, -a_clip, a_clip)
            st["prev_goal"] = np.array(st["goal"], dtype=float)
        st["goal"] = q.copy()
        st["v_ref"] = v_new
        st["a_ref"] = a_new
        st["ref_start"] = np.array(st["prev_goal"], dtype=float)

    def run_controller():
        if ctrl.goal_qpos is None:
            abs_set_goal(np.zeros(dim))
        ctrl.update()
        desired = np.array(ctrl.goal_qpos, dtype=float)
        tau = float(np.clip((float(ctrl.sim.data.time) - st["t_step"]) / dt, 0.0, 1.0))
        ref = st["ref_start"] + (desired - st["ref_start"]) * tau
        des_acc = (
            ctrl.kp * (ref - ctrl.joint_pos)
            + ctrl.kd * (st["v_ref"] - ctrl.joint_vel)
            + st["a_ref"]
        )
        ctrl.torques = np.dot(ctrl.mass_matrix, des_acc) + ctrl.torque_compensation
        # ★ 基类 run_controller 的唯一副作用是把 new_update 置回 True，
        #   漏掉它控制器会一直用第一次读到的状态（实测误差发散到 2-3 rad）。
        _BaseController.run_controller(ctrl)
        return ctrl.torques

    ctrl.set_goal = abs_set_goal
    ctrl.run_controller = run_controller
    ctrl._lab_state = st
    ctrl._joint_protocol = "v1"


def attach(robot, protocol: str, dt: float) -> None:
    """按协议名给一个 robot 的 controller 打补丁（未知协议名直接报错，别静默退回 v0）。"""
    if protocol == "v0":
        patch_absolute_set_goal(robot)
    elif protocol == "v1":
        attach_reference_feedforward(robot, dt=dt)
    else:
        raise ValueError(f"unknown joint_protocol {protocol!r}; expected one of {JOINT_PROTOCOLS}")


def reset_protocol_state(robot) -> None:
    """清空 v1 的每回合状态（v0 无状态，直接跳过）。

    必须在每次 `env.reset()` 之后调用：LIBERO 会重建 controller，新对象本来没有状态；
    但如果 controller 被复用（软 reset），不清空会让第一步的 `v_ref`/`a_ref` 带着上一回合的残值。
    """
    st = getattr(robot.controller, "_lab_state", None)
    if not st:
        return
    dim = _control_dim(robot.controller)
    st.update(
        {
            "goal": None,
            "prev_goal": None,
            "v_ref": np.zeros(dim),
            "a_ref": np.zeros(dim),
            "t_step": 0.0,
            "ref_start": None,
            "n_goals": 0,
        }
    )
