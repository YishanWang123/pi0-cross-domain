# eval_robomimic_websocket.py
import collections
import dataclasses
import datetime
import json
import logging
import math
import os
import pathlib
from typing import Dict, List, Optional

import imageio
import numpy as np
import pytz
import tqdm
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# ----- robomimic imports -----
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.dataset import ObsUtils


ROBOMIMIC_RESIZE = 224


# robomimic task
TASK_TO_LANG: Dict[str, str] = {
    "lift": "lift the small red cube",
    "can": "place a coke can from a large bin into a smaller target bin",
    "square": "pick a square nut and place it on a rod",
    "tool_hang": "assemble a frame and hang a wrench on the hook",
    # "transport": "use two robot arms to transfer a hammer from a closed container to a target bin",
}

# Max steps for each task
TASK_MAX_STEPS = {
    "lift": 400,
    "can": 200,
    "square": 400,
    "tool_hang": 650,
    "transport": 800,
}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """从 robosuite 抄来的四元数→轴角（与 LIBERO 保持一致）"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def build_state_vec(obs):
    """返回 state 向量 (单臂7维 / 双臂14维) 和手臂数 1/2"""
    def ee_state(prefix):
        return np.concatenate([
            obs[f"{prefix}_eef_pos"],                     # 3
            _quat2axisangle(obs[f"{prefix}_eef_quat"]),   # 3
            obs[f"{prefix}_gripper_qpos"]                 # 1
        ], axis=0)                                        # -> 7

    two_arms = ("robot1_eef_pos" in obs)
    if two_arms:
        s0 = ee_state("robot0")
        s1 = ee_state("robot1")
        return np.concatenate([s0, s1], axis=0), 2        # (14,), 2 arms
    else:
        return ee_state("robot0"), 1                      # (7,), 1 arm

def _ensure_uint8_image(img: np.ndarray) -> np.ndarray:
    """
    robomimic 里观测可能是 [0,1] float 或 uint8，这里统一成 uint8。
    同时做 letterbox 到 ROBOMIMIC_RESIZE（与 LIBERO 脚本保持一致的预处理）。
    """
    if img.dtype != np.uint8:
        # 常见：float32 的 [0,1]
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, ROBOMIMIC_RESIZE, ROBOMIMIC_RESIZE)
    )
    return img


def _build_env_from_dataset(dataset_path: str, use_image_obs: bool = True):
    """
    根据 robomimic 数据集元信息构建环境（离线仿真评测）
    - 将 controller_configs.control_delta 设为 True（通常 LIBERO 风格策略输出的是“增量”位姿）
      如果你的 server 输出绝对位姿动作，可以改回 False。
    """
    ObsUtils.initialize_obs_modality_mapping_from_dict({
        "low_dim": [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"
        ],
        "rgb": ["robot0_eye_in_hand_image", "agentview_image"],
        "depth": [],
        "scan": [],
    })

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    # 尝试使用增量控制以匹配常见策略输出（如 LIBERO）
    try:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = True
    except Exception:
        pass  # 若不存在该字段则跳过

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=False,
        use_image_obs=use_image_obs,
    )
    return env


# ===============================
# 配置参数（仿照 LIBERO 脚本）
# ===============================

@dataclasses.dataclass
class Args:
    # -----------------------------
    # Model server parameters
    # -----------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 20

    # -----------------------------
    # robomimic 评测参数
    # -----------------------------
    tasks: List[str] = dataclasses.field(default_factory=lambda: ["tool_hang"])
    num_trials_per_task: int = 5
    num_steps_wait: int = 0  # robomimic 一般无需等待“落物”，保留开关

    # robomimic 数据集根目录（按 robomimic 官方组织方式）
    dataset_root: str = "/mnt/ssd0/data/robomimic/env_hdf5"

    # -----------------------------
    # 复现 & 输出
    # -----------------------------
    seed: int = 7
    video_out_path: str = "data/robomimic/videos"
    result_json_path: str = "results_robomimic_websocket.json"


# ===============================
# 主评测逻辑
# ===============================

def eval_robomimic(args: Args) -> None:
    np.random.seed(args.seed)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    task_results = []

    for task_name in tqdm.tqdm(args.tasks, desc="Tasks"):
        
        # if task_name.lower() == "transport":
        #     logging.warning("Task 'transport' (two-arm) is not supported in this single-arm script. Skipping.")
        #     continue

        max_steps = TASK_MAX_STEPS.get(task_name, 500)

        # 构造数据集路径（与常见 robomimic 目录一致：{root}/{task}/ph/image.hdf5）
        dataset_path = os.path.join(args.dataset_root, task_name, "ph", "image.hdf5")
        if not os.path.exists(dataset_path):
            logging.error(f"[{task_name}] dataset not found at: {dataset_path}")
            continue

        # 初始化环境
        env = _build_env_from_dataset(dataset_path, use_image_obs=True)

        # 语言
        task_prompt = TASK_TO_LANG.get(task_name, f"do task: {task_name}")

        task_episodes, task_successes = 0, 0

        for ep in tqdm.tqdm(range(args.num_trials_per_task), desc=f"{task_name} episodes", leave=False):
            obs = env.reset()
            print(f"obs keys: {obs.keys()}")
            action_plan = collections.deque()
            state_vec, num_arms = build_state_vec(obs)
            dummy = ([0.0] * 6 + [-1.0]) if num_arms == 1 else ([0.0] * 6 + [-1.0]) * 2

            replay_images = []
            t = 0
            while t < max_steps + args.num_steps_wait:
                try:
                    # 可选：等待若干步（robomimic 通常不需要）
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(dummy)
                        t += 1
                        continue

                    # 取两路相机并做统一预处理
                    main_img = obs["agentview_image"]
                    wrist_img = obs["robot0_eye_in_hand_image"]

                    # 标准化到 224x224 uint8
                    main_img = _ensure_uint8_image(main_img)
                    wrist_img = _ensure_uint8_image(wrist_img)

                    # 记录回放帧（这里拼接成单帧：左主视角，右腕部）
                    replay_images.append(np.concatenate([main_img, wrist_img], axis=1))

                    # 计划用尽 → 触发重规划
                    if not action_plan:
                        state_vec, num_arms = build_state_vec(obs)

                        element = {
                            "observation/image": main_img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state_vec,
                            "prompt": str(task_prompt),
                        }

                        print(element["observation/state"].shape)
                    # 取一步动作并执行
                        action_chunk = client.infer(element)["actions"]
                        print(f"action_chunk shape: {action_chunk.shape}")
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    success = env.is_success()["task"]
                    if done or success:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1


                except Exception as e:
                    logging.error(f"[{task_name}] exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # 保存视频
            suite_folder = pathlib.Path(args.video_out_path) / task_name
            suite_folder.mkdir(parents=True, exist_ok=True)
            suffix = "success" if (done or success) else "failure"
            tz_sh = pytz.timezone("Asia/Shanghai")
            time_str = datetime.datetime.now(tz_sh).strftime("%Y%m%d_%H%M%S")
            video_path = suite_folder / f"rollout_{task_name}_{suffix}_{time_str}.mp4"
            if replay_images:
                imageio.mimwrite(str(video_path), [np.asarray(x) for x in replay_images], fps=10)

            logging.info(f"[{task_name}] ep#{task_episodes} success={suffix}")

        # 记录任务级别结果
        sr = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        logging.info(f"[{task_name}] success rate: {sr:.3f} ({task_successes}/{task_episodes})")
        task_results.append({
            "task": task_name,
            "success_rate": sr,
            "episodes": task_episodes
        })

    # 汇总与保存
    total_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    logging.info(f"[robomimic] Total success rate: {total_sr:.3f} ({total_successes}/{total_episodes})")
    result_data = {
        "suite": "robomimic_single_arm",
        "total_success_rate": total_sr,
        "total_episodes": total_episodes,
        "tasks": task_results,
    }

    # 循环结束后写总表，按 task 名归档
    summary_path = pathlib.Path("results_robomimic_summary.json")
    summary = {}

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    for item in task_results:
        summary[item["task"]] = {
            "success_rate": item["success_rate"],
            "episodes": item["episodes"],
            "updated_at": datetime.datetime.now().isoformat()
        }

    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_robomimic)