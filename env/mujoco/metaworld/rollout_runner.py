# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import functools

import cv2
import numpy as np
from .test_scripted_policies import (
    ALL_ENVS,
    test_cases_latest_nonoise,
)

import torch
from collections import OrderedDict
from tqdm import tqdm
import traceback
import json

ALL_TASK_CONFIG = [
    ("assembly-v3", np.zeros(4), 10000, True, "pick up the pole and put the circle through the cylinder."),
    ("basketball-v3", np.zeros(4), 10000, True, "pick up the basketball and drop it through the basket."),
    ("bin-picking-v3", np.zeros(4), 10000, True, "move the block from one bin to another bin."),
    ("box-close-v3", np.zeros(4), 10000, True, "close the lid of the box."),
    ("button-press-topdown-v3", np.zeros(4), 10000, True, "press down the button."),
    ("button-press-topdown-wall-v3", np.zeros(4), 10000, True, "press down the button with one finger."),
    ("button-press-v3", np.zeros(4), 10000, True, "press in the button with one finger."),
    ("button-press-wall-v3", np.zeros(4), 100, True, "move between wall and botton and open fingers."),
    ("coffee-button-v3", np.zeros(4), 10000, True, "touch the coffee mug."),
    ("coffee-pull-v3", np.zeros(4), 10000, True, "grasp the coffee mug."),
    ("coffee-push-v3", np.zeros(4), 10000, True, "push the coffee mug."),
    ("dial-turn-v3", np.zeros(4), 10000, True, "turn the dial with the fingers."),
    ("disassemble-v3", np.zeros(4), 10000, True, "pull the circle bar out of the cylinder."),
    ("door-close-v3", np.zeros(4), 10000, True, "close the door with the fingers."),
    ("door-lock-v3", np.zeros(4), 10000, True, "lock the door."),
    ("door-open-v3", np.zeros(4), 10000, True, "open the door."),
    ("door-unlock-v3", np.zeros(4), 10000, True, "unlock the door."),
    ("hand-insert-v3", np.zeros(4), 10000, True, "pick up the wooden block and put it in the box."),
    ("drawer-close-v3", np.zeros(4), 10000, True, "close the drawer with the fingers."),
    ("drawer-open-v3", np.zeros(4), 10000, True, "open the drawer with the fingers."),
    ("faucet-open-v3", np.zeros(4), 10000, True, "turn the faucet to open."),
    ("faucet-close-v3", np.zeros(4), 10000, True, "turn the faucet to close."),
    ("hammer-v3", np.zeros(4), 10000, True, "grasp the hammer and move towards the button."),
    ("handle-press-side-v3", np.zeros(4), 10000, True, "press the handle down."),
    ("handle-press-v3", np.zeros(4), 10000, True, "use the side finger to press the handle."),
    ("handle-pull-side-v3", np.zeros(4), 10000, True, "pull the handle"),
    ("handle-pull-v3", np.zeros(4), 10000, True, "pull the handle."),
    ("lever-pull-v3", np.zeros(4), 10000, True, "pull the lever."),
    ("pick-place-wall-v3", np.zeros(4), 10000, True, "pick the red cylinder and place it at the blue spot."),
    ("pick-out-of-hole-v3", np.zeros(4), 10000, True, "pick up the red cylinder."),
    ("push-back-v3", np.zeros(4), 10000, True, "move the wooded brick to the green spot."),
    ("push-v3", np.zeros(4), 10000, True, "push the red cylinder to the green spot."),
    ("pick-place-v3", np.zeros(4), 10000, True, "pick up the red cylinder and put it in the blue spot."),
    ("plate-slide-v3", np.zeros(4), 10000, True, "pick up the gray cylinder and move it to the red bucket."),
    ("plate-slide-side-v3", np.zeros(4), 10000, True, "push the gray cylinder into the red bucket"),
    ("plate-slide-back-v3", np.zeros(4), 10000, True, "move the gray cylinder out of the red bucket"),
    ("plate-slide-back-side-v3", np.zeros(4), 10000, True, "pull the gray cylinder out of the red bucket"),
    ("peg-insert-side-v3", np.zeros(4), 10000, True, "insert the green peg into the red wall."),
    ("peg-unplug-side-v3", np.zeros(4), 10000, True, "unplug the gray cylinder out."),
    ("soccer-v3", np.zeros(4), 10000, True, "push the soccer ball into the goal net."),
    ("stick-push-v3", np.zeros(4), 10000, True, "pick up the blue box and push the gray."),
    ("stick-pull-v3", np.zeros(4), 10000, True, "pick the blue box and pull the gray."),
    ("push-wall-v3", np.zeros(4), 10000, True, "move the red cylinder around the wall to the green spot."),
    ("reach-wall-v3", np.zeros(4), 10000, True, "reach the red spot on the wall."),
    ("reach-v3", np.zeros(4), 10000, True, "reach the wall."),
    ("shelf-place-v3", np.zeros(4), 10000, True, "put the blue block inside the shelf."),
    ("sweep-into-v3", np.zeros(4), 10000, True, "pick up the wooden block and put inside the hole."),
    ("sweep-v3", np.zeros(4), 10000, True, "pick the wooden block and then drop it."),
    ("window-open-v3", np.zeros(4), 10000, True, "open the window by pushing the handle."),
    ("window-close-v3", np.zeros(4), 10000, True, "close the window by pulling the handle."),
]
RESOLUTION = (128, 128)


def writer_for(tag, fps, res, src_folder="demonstrations"):
    if not os.path.exists(src_folder):
        os.mkdir(src_folder)
    return cv2.VideoWriter(
        f"{src_folder}/{tag}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        res,
    )


@torch.no_grad()
def learner_trajectory_generator(env, policy, lang="", camera_name="view_1"):
    """generate a trajectory rollout from a policy and a metaworld environment"""
    env.reset()
    env.reset_model()
    policy.reset()
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        o = reset_result[0]
    else:
        o = reset_result
    env.render_mode = 'rgb_array'
    img = env.render()[:, :, ::-1].copy()

    def get_observation_dict(o, img):
        step_data = {"state": o, "image": img}
        return OrderedDict(step_data)

    step_data = get_observation_dict(o, img)
    for _ in range(env.max_path_length):
        a = policy.get_action(step_data)
        step_result = env.step(a)
        if len(step_result) == 5:
            o, r, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            o, r, done, info = step_result

        img = env.render()[:, :, ::-1]
        img = cv2.resize(img, RESOLUTION).astype(np.uint8)

        ret = [o, r, done, info, img]
        step_data = get_observation_dict(o, img)
        yield ret


class RolloutRunner:
    """evaluate policy rollouts"""

    def __init__(self, env_names, episode_num, save_video=False):
        self.env_names = env_names
        self.episode_num = episode_num
        self.save_video = save_video

    @torch.no_grad()
    def run(
        self,
        policy,
        save_video=False,
        gui=False,
        video_postfix="",
        video_path=None,
        env_name=None,
        seed=233,
        episode_num=-1,
        **kwargs,
    ):
        camera = "view_1"
        flip = True
        noise = np.zeros(4)
        quit_on_success = True
        if episode_num == -1:
            episode_num = self.episode_num
        all_env_names = [task for (task, _, _, _, lang) in ALL_TASK_CONFIG]
        env_lang_map = {task: lang for (task, _, _, _, lang) in ALL_TASK_CONFIG}

        if type(self.env_names) is not list:
            env_names = all_env_names
        else:
            env_names = self.env_names.split(",")

        for env in env_names:
            env = env.strip()
            if env not in all_env_names:
                continue

            language_instruction = [task for (task, _, _, _, lang) in ALL_TASK_CONFIG if task == env][0]
            if env_name is not None:
                env_name = env_name
                if str(env_name[0]) != str(env):
                    continue

            print("env_name:", env_name, episode_num)
            tag = env
            env_keys = sorted(list(ALL_ENVS.keys()))
            env = ALL_ENVS[env]()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.seed(seed)

            if self.save_video:
                writer = writer_for(
                    tag + f"_{video_postfix}",
                    env.metadata["video.frames_per_second"],
                    RESOLUTION,
                    src_folder="output/output_figures/output_videos/metaworld",
                )

            total_success = 0
            total_reward = 0
            pbar = tqdm(range(episode_num), position=1, leave=True)
            try:
                for i in pbar:
                    eps_reward = 0
                    traj_length = 0
                    q_pos = []

                    step = 0
                    for o, r, done, info, img in learner_trajectory_generator(env, policy, language_instruction):
                        traj_length += 1
                        eps_reward += r
                        if self.save_video and i <= 5:
                            if gui:
                                cv2.imshow("img", img)
                                cv2.waitKey(1)
                            writer.write(img)

                        if info["success"]:
                            break

                        step += 1
                    pbar.set_description(f"success: {info['success']}")
                    total_success += info["success"]
                    total_reward += eps_reward
            except Exception as e:
                print(traceback.format_exc())
            return total_success / episode_num, total_reward / episode_num


@torch.no_grad()
def expert_trajectory_generator(env, policy, camera_name="view_1"):
    """generate a trajectory rollout from a policy and a metaworld environment"""
    env.reset()
    env.reset_model()
    env.render_mode = 'rgb_array'
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        o = reset_result[0]
    else:
        o = reset_result
    for _ in range(env.max_path_length):
        a = policy.get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        step_result = env.step(a)
        if len(step_result) == 5:
            o, r, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            o, r, done, info = step_result

        img = env.render()[:, :, ::-1]
        img = cv2.resize(img, RESOLUTION).astype(np.uint8)
        ret = [a, o, r, done, info, img]
        yield ret


def generate_dataset_rollouts(
    env_names, save_video=False, gui=False, max_total_transition=2000, episode_num_pertask=100, **kwargs
):
    """online generate scripted expert data for a env"""
    camera = "view_1"
    flip = True
    noise = np.zeros(4)
    quit_on_success = True
    cycles = episode_num_pertask // len(env_names)
    all_env_names = [task for (task, _, _, _, lang) in ALL_TASK_CONFIG]

    if env_names == "all":
        env_names = all_env_names
    else:
        env_names = list(env_names)

    print("metaworld env names:", env_names)
    for env in env_names:
        env = env.strip()
        if env not in all_env_names:
            continue

        language_instruction = [task for (task, _, _, _, lang) in ALL_TASK_CONFIG if task == env][0]
        tag = env
        policy = functools.reduce(lambda a, b: a if a[0] == env else b, test_cases_latest_nonoise)[1]
        env_keys = sorted(list(ALL_ENVS.keys()))
        env = ALL_ENVS[env]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True

        if save_video:
            writer = writer_for(tag, env.metadata["video.frames_per_second"], RESOLUTION)

        dataset_traj_states = []
        dataset_traj_actions = []
        dataset_traj_images = []
        total_transition_num = 0

        for i in range(cycles):
            eps_reward = 0
            traj_length = 0
            eps_states = []
            eps_actions = []
            eps_images = []
            q_pos = []

            step = 0
            try:
                for a, o, r, done, info, img in expert_trajectory_generator(env, policy):
                    eps_states.append(o)
                    eps_actions.append(a)
                    eps_images.append(img)
                    traj_length += 1
                    eps_reward += info["success"]
                    if save_video and i <= 10:
                        if gui:
                            cv2.imshow("img", img)
                            cv2.waitKey(1)
                        writer.write(img)

                    if info["success"]:
                        break

                    step += 1
            except:
                print(traceback.format_exc())
                continue

            print("success:", info["success"])
            if info["success"]:
                total_transition_num += len(eps_images)
            else:
                total_transition_num += len(eps_images)

            print(f"data generation number of episodes: {tag} {i} {total_transition_num}")
            steps = []

            eps_actions = eps_actions[1:]
            for state, action, image in zip(eps_states, eps_actions, eps_images):
                step = {
                    "action": action,
                    "observation": {"state": state, "image": image}
                }
                steps.append(step)
            data_dict = {"steps": steps}
            yield data_dict


if __name__ == "__main__":
    runner = RolloutRunner(["all"], 200)
