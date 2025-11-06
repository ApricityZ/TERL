import copy
import json
import os
import pdb
import sys
import time
from typing import Optional, Any, List, Tuple

import gym
import numpy as np
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from scipy.spatial import KDTree

import utils
from config_manager import ConfigManager
from robots.evader import Evader
from robots.pursuer import Pursuer
from utils import logger as logger


class Core:
    """
    Initialize a vortex core object.

    Attributes:
        x (np.ndarray): X-coordinates of vortex core.
        y (np.ndarray): Y-coordinates of vortex core.
        clockwise (bool): Vortex rotation direction (True for clockwise).
        Gamma (float): Circulation strength of vortex core.
    """

    def __init__(self, x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]],
                 clockwise: bool, gamma: float):
        """
        Args:
            x (np.ndarray): X-coordinates of vortex core.
            y (np.ndarray): Y-coordinates of vortex core.
            clockwise (bool): Vortex rotation direction.
            gamma (float): Circulation strength of vortex core.
        """
        self.x = x  # x coordinate of the vortex core
        self.y = y  # y coordinate of the vortex core
        self.clockwise = clockwise  # if the vortex direction is clockwise
        self.Gamma = gamma  # circulation strength of the vortex core


class Obstacle:
    """
    Initialize an obstacle object.

    Attributes:
        x (np.ndarray): X-coordinates of obstacle center.
        y (np.ndarray): Y-coordinates of obstacle center.
        r (float): Radius of obstacle.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, r: float):
        """
        Args:
            x (np.ndarray): X-coordinates of obstacle center.
            y (np.ndarray): Y-coordinates of obstacle center.
            r (float): Radius of obstacle.
        """
        self.x = x  # x coordinate of obstacle center
        self.y = y  # y coordinate of obstacle center
        self.r = r  # obstacle radius


class MarineEnv(gym.Env):
    r"""
    Marine environment class for reinforcement learning tasks.

    Simulates a marine environment with dynamic/static obstacles including vortex cores and static obstacles.
    Supports curriculum learning with environmental changes across multiple episodes.

    Attributes:
        sd (int): Random seed.
        rd (np.random.RandomState): Random number generator.
        width (int): Map x-axis dimension.
        height (int): Map y-axis dimension.
        vortex_core_radius (float): Vortex core radius.
        v_rel_max (float): Max relative velocity between opposing currents.
        p (float): Max allowed relative velocity between vortex cores.
        v_range (list): Velocity range at vortex edge.
        obs_r_range (list): Obstacle radius range.
        clear_r (float): Safe radius around robot start/goal positions.
        min_pursuer_evader_init_dis (float): Min start distance between pursuers and evaders.
        min_evader_evader_init_dis (float): Min start distance between evaders.
        timestep_penalty (float): Penalty per timestep.
        distance_reward (float): Distance reward value.
        collision_penalty (float): Collision penalty value.
        goal_reward (float): Goal achievement reward.
        num_cores (int): Number of vortex cores.
        num_obs (int): Number of obstacles.
        num_pursuers (int): Number of pursuers.
        num_evaders (int): Number of evaders.
        pursuers (List[Pursuer]): List of pursuers.
        evaders (List[Evader]): List of evaders.
        cores (list): List of vortex cores.
        core_centers (None): Vortex core positions.
        obstacles (list): List of obstacles.
        schedule (dict): Curriculum learning schedule.
        episode_time_steps (int): Current episode timesteps.
        episode_max_length (int): Max episode length.
        total_time_steps (int): Total training timesteps.
        observation_in_robot_frame (bool): Return observations in robot frame.
    """

    def __init__(self, seed: int = 0, schedule: dict = None):
        """
        Initialize marine environment object.

        Args:
            seed (int): Random seed. Default 0.
            schedule (dict): Curriculum learning schedule. Default None.
        """

        self.sd = seed
        self.rd = np.random.RandomState(seed)  # PRNG

        # Parameter initialization
        # Map parameters
        self.config_env = ConfigManager().get_instance()
        self.width = self.config_env.get("env.width", default=100)  # x-axis dimension
        self.height = self.config_env.get("env.height", default=100)  # y-axis dimension
        self.arena_size = self.width
        # Vortex parameters
        self.vortex_core_radius = 0.5  # vortex core radius
        self.v_rel_max = 1.0  # max relative velocity between opposing currents
        self.p = 0.8  # max allowed relative velocity between vortex cores
        self.v_range = [5, 10]  # velocity range at vortex edge

        # Obstacle parameters
        self.obs_r_range = self.config_env.get("env.obs_r_range", default=[1.0, 1.0])  # obstacle radius range

        # Robot generation parameters
        self.clear_r = self.config_env.get("env.clear_r", default=15.0)  # safe zone around start positions
        self.min_pursuer_evader_init_dis = 15.0  # min pursuer-evader start distance
        self.min_evader_evader_init_dis = 10.0  # min evader-evader start distance

        # Reward/penalty parameters
        self.goal_reward = self.config_env.get("env.goal_reward", default=100.0)  # goal achievement reward
        self.distance_reward = self.config_env.get("env.distance_reward", default=5.0)  # distance reward
        self.timestep_penalty = self.config_env.get("env.timestep_penalty", default=-2.0)  # per-timestep penalty

        # Additional penalties
        self.emergency_penalty = self.config_env.get("env.emergency_penalty", default=-5.0)
        self.collision_penalty = self.config_env.get("env.collision_penalty", default=-80.0)
        self.boundary_penalty = self.config_env.get("env.boundary_penalty", default=-5.0)
        self.decay_factor = self.config_env.get("env.decay_factor", default=-0.05)

        # Global cooperation rewards
        self.evenly_distributed_reward = self.config_env.get("env.evenly_distributed_reward",
                                                             default=5.0)  # optimal pursuer distribution reward
        self.unevenly_distributed_reward = self.config_env.get("env.unevenly_distributed_reward",
                                                               default=-10.0)  # excessive clustering penalty

        # Distance parameters
        self.capture_distance = self.config_env.get("env.capture_distance", default=8.0)  # capture distance
        self.related_distance = self.config_env.get("env.related_distance", default=18.0)  # cooperation initiation distance

        # Entity counts
        self.num_cores = 8  # number of vortex cores
        self.num_obs = 8  # number of obstacles
        self.num_pursuers = 1  # number of pursuers
        self.num_evaders = 1  # number of evaders

        # Entity lists
        self.pursuers: List[Pursuer] = []  # pursuer list
        self.evaders: List[Evader] = []  # evader list
        self.pursuers.append(Pursuer(0))
        self.evaders.append(Evader(0))

        # Environment parameters
        self.cores: List[Core] = []  # vortex core list
        self.core_centers = None
        self.obstacles: List[Obstacle] = []  # obstacle list

        # Curriculum learning parameters
        self.schedule = schedule  # curriculum schedule
        self.episode_time_steps = 0  # current episode timesteps
        self.episode_max_length = self.config_env.get("training.episode_max_length", default=3000)  # max episode length
        self.total_time_steps = 0  # total training timesteps

        # Observation parameters
        self.observation_in_robot_frame = True  # return observations in robot frame

    def get_action_space_dimension(self) -> int:
        """
        Get pursuer action space dimension.

        Returns:
            int: Action space dimension.
        """
        return self.pursuers[0].compute_actions_dimension()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment.

        Resets environment according to curriculum schedule, including regenerating:
        pursuers, evaders, vortex cores, and obstacles.

        Args:
            seed (Optional[int]): Random seed. Default None.
            options (Optional[dict]): Reset options. Default None.

        Returns:
            list: Pursuer observations.
        """
        if self.schedule is not None:
            steps = self.schedule["time_steps"]  # curriculum phase timesteps
            diffs = np.array(steps) - self.total_time_steps  # phase difference

            # Determine current curriculum phase
            idx = len(diffs[diffs <= 0]) - 1  # phase index

            self.num_pursuers = self.schedule["num_pursuers"][idx]
            self.num_evaders = self.schedule["num_evaders"][idx]
            self.num_cores = self.schedule["num_cores"][idx]
            self.num_obs = self.schedule["num_obstacles"][idx]
            self.min_pursuer_evader_init_dis = self.schedule["min_pursuer_evader_init_dis"][idx]

            logger.info(f"Process {os.getpid()} ======== training schedule ========")
            logger.info(f"Process {os.getpid()} num of pursuers: {self.num_pursuers}")
            logger.info(f"Process {os.getpid()} num of evaders: {self.num_evaders}")
            logger.info(f"Process {os.getpid()} num of cores: {self.num_cores}")
            logger.info(f"Process {os.getpid()} num of obstacles: {self.num_obs}")
            logger.info(f"Process {os.getpid()} min pursuer_start goal dis: {self.min_pursuer_evader_init_dis}")
            logger.info(f"Process {os.getpid()} ======== training schedule ========\n")

        self.episode_time_steps = 0

        self.cores.clear()
        self.obstacles.clear()
        self.pursuers.clear()
        self.evaders.clear()

        # Generate evaders first
        num_evaders = 0
        iteration = 50000
        while True:
            # Evader start positions away from boundaries
            evader_start = self.rd.uniform(low=15.0 * np.ones(2),
                                           high=np.array([self.width - 15.0, self.height - 15.0]))
            iteration -= 1
            if self.check_evader_start_position(evader_start=evader_start):
                evader = Evader(index=num_evaders)
                evader.start = evader_start
                self.reset_robot(evader)
                self.evaders.append(evader)
                num_evaders += 1
            if iteration == 0 or num_evaders == self.num_evaders:
                if num_evaders == self.num_evaders:
                    break
                else:
                    logger.error("Failed to generate required number of evaders🥲!")
                    sys.exit("-1 at code:reset, MarineEnv, line 299. Failed to generate required number of evaders🥲!")

        # Generate pursuers
        num_pursuers = 0
        iteration = 8000
        while True:
            # Pursuer start positions
            pursuer_start = self.rd.uniform(low=2.0 * np.ones(2),
                                            high=np.array([self.width - 2.0, self.height - 2.0]))
            iteration -= 1
            if self.check_pursuer_start_position(pursuer_start):
                pursuer = Pursuer(index=num_pursuers)
                pursuer.start = pursuer_start
                self.reset_robot(pursuer)
                self.pursuers.append(pursuer)
                num_pursuers += 1
            if iteration == 0 or num_pursuers == self.num_pursuers:
                if num_pursuers == self.num_pursuers:
                    break
                else:
                    logger.error("Failed to generate required number of pursuers🥲!")
                    sys.exit("-1 at code:reset, MarineEnv, line 322. Failed to generate required number of pursuers🥲!")

        # Generate vortex cores
        num_cores = self.num_cores
        if num_cores > 0:
            iteration = 8000
            while True:
                # Random vortex core center
                center = self.rd.uniform(low=np.zeros(2), high=np.array([self.width, self.height]))
                # Random rotation direction
                direction = self.rd.binomial(1, 0.5)
                # Reduced intensity parameters
                v_edge = self.rd.uniform(low=self.v_range[0] * 0.5, high=self.v_range[1] * 0.5)
                gamma = 2 * np.pi * (self.vortex_core_radius * 0.5) * v_edge
                core = Core(center[0], center[1], bool(direction), gamma)
                iteration -= 1
                if self.check_core(core):
                    self.cores.append(core)
                    num_cores -= 1
                if iteration == 0 or num_cores == 0:
                    if len(self.cores) == self.num_cores:
                        break
                    else:
                        logger.error(f"Failed to generate required number of {len(self.cores)}/{self.num_cores} cores🥲!")
                        sys.exit("-1 at code:reset, MarineEnv, line 347. Failed to generate required number of cores🥲!")

        centers = None
        for core in self.cores:
            if centers is None:
                centers = np.array([[core.x, core.y]])
            else:
                c = np.array([[core.x, core.y]])
                centers = np.vstack((centers, c))

        # Store core positions in KDTree
        if centers is not None:
            self.core_centers = KDTree(centers)

        # Generate obstacles
        num_obs = self.num_obs
        if num_obs > 0:
            iteration = 8000
            while True:
                center = self.rd.uniform(low=5.0 * np.ones(2), high=np.array([self.width - 5.0, self.height - 5.0]))
                r = self.rd.uniform(low=self.obs_r_range[0], high=self.obs_r_range[1])
                obs = Obstacle(center[0], center[1], r)
                iteration -= 1
                if self.check_obstacle(obs):
                    self.obstacles.append(obs)
                    num_obs -= 1
                if iteration == 0 or num_obs == 0:
                    if len(self.obstacles) == self.num_obs:
                        break
                    else:
                        logger.error(f"Failed to generate required number {len(self.obstacles)}/{self.num_obs} of obstacles🥲!")
                        sys.exit("-1 at code:reset, MarineEnv, line 377. Failed to generate required number of obstacles🥲!")

        return self.get_pursuers_observations(), self.get_pursuers_captured_state()

    def assign_robot_position(self, pursuer_position_list: List[np.ndarray], evader_position_list: List[np.ndarray],
                              pursuer_theta_list: List[float] = None, evader_theta_list: List[float] = None):
        """
        Assign robot positions.

        Args:
            pursuer_position_list (List[np.ndarray]): Pursuer positions.
            evader_position_list (List[np.ndarray]): Evader positions.
            pursuer_theta_list (List[float]): Pursuer orientations.
            evader_theta_list (List[float]): Evader orientations.
        """
        try:
            assert len(pursuer_position_list) == len(self.pursuers), ("Number of pursuers should be equal to number of "
                                                                      "pursuers!")
            assert len(evader_position_list) == len(self.evaders), ("Number of evaders should be equal to number of "
                                                                    "evaders!")
        except AssertionError as e:
            logger.info(f"Process {os.getpid()} len(pursuer_position_list): {len(pursuer_position_list)}, "
                        f"len(self.pursuer): {len(self.pursuers)}, "
                        f"len(evader_position_list): {len(evader_position_list)}, "
                        f"len(self.evaders): {len(self.evaders)}")
            logger.error(f"AssertionError: {e}.")
            sys.exit(f"Error: {e}")

        for i, _pursuer in enumerate(self.pursuers):
            _pursuer.start = pursuer_position_list[i]
            self.reset_robot(_pursuer)
        for i, _evader in enumerate(self.evaders):
            _evader.start = evader_position_list[i]
            self.reset_robot(_evader)
        if pursuer_theta_list is not None:
            for i, _pursuer in enumerate(self.pursuers):
                _pursuer.theta = pursuer_theta_list[i]
        if evader_theta_list is not None:
            for i, _evader in enumerate(self.evaders):
                _evader.theta = evader_theta_list[i]



    def reset_robot(self, rob: Pursuer | Evader):
        """
        Reset robot state.

        Resets the robot's initial state including position, velocity, and perception data.

        Args:
            rob (Pursuer | Evader): Robot instance to reset.
        """
        if rob.robot_type == 'pursuer':
            rob.is_current_target_captured = False
        rob.collision = False
        rob.deactivated = False
        rob.init_theta = self.rd.uniform(low=0.0, high=2 * np.pi)
        rob.init_speed = self.rd.uniform(low=0.0, high=rob.max_speed)
        current_v = self.get_current_velocity(rob.start[0], rob.start[1])
        rob.reset_state(current_velocity=current_v)

    def check_all_pursuers_deactivated(self) -> bool:
        """
        Check if all pursuers are deactivated.

        Returns:
            bool: True if all pursuers are deactivated, False otherwise.
        """
        return all(rob.deactivated for rob in self.pursuers)

    def check_all_evaders_deactivated(self) -> bool:
        """
        Check if all evaders are deactivated.

        Returns:
            bool: True if all evaders are deactivated, False otherwise.
        """
        return all(rob.deactivated for rob in self.evaders)

    def check_all_evader_is_captured(self) -> bool:
        """
        Check if all evaders have been captured.

        Returns:
            bool: True if all evaders are captured, False otherwise.
        """
        return all(evader.deactivated for evader in self.evaders)

    def check_any_pursuers_collision(self) -> bool:
        """
        Check if any pursuer has collided.

        Returns:
            bool: True if any pursuer has collided, False otherwise.
        """
        return any(rob.collision for rob in self.pursuers)

    def check_any_evaders_collision(self) -> bool:
        """
        Check if any evader has collided.

        Returns:
            bool: True if any evader has collided, False otherwise.
        """
        return any(rob.collision for rob in self.evaders)

    def update_evader_state(self, evader_actions: List[int]):
        """
        Update evader states.

        Updates evader states including position, velocity and trajectory based on actions.

        Args:
            evader_actions (List[int]): List of actions for evaders.
        """
        for i, evader_action in enumerate(evader_actions):
            evader = self.evaders[i]

            if evader.deactivated:
                continue

            evader.action_history.append(evader_action)

            self.update_rob_state(evader, evader_action)

            self.update_pursuing_status()

            evader.trajectory.append([evader.x, evader.y, evader.theta, evader.speed, evader.velocity[0],
                                      evader.velocity[1]])

    def get_pursuers_distance_evaders(self) -> List[List[float] | None]:
        """
        Get distances between pursuers and their targets.

        Returns:
            List[List]: List of distances from pursuers to their targets.
        """
        return self.get_distance_to_evaders()

    def get_distance_to_evaders(self) -> List[Optional[float]]:
        """
        Get distances from pursuers to evaders.

        Returns:
            List[Optional[float]]: List of minimum distances from each pursuer to all evaders.
        """
        all_pursuers_distance_to_evaders = []
        for pursuer in self.pursuers:
            if pursuer.deactivated:
                all_pursuers_distance_to_evaders.append(None)
                continue
            pursuer_distance_to_evaders = [
                np.linalg.norm([pursuer.x - evader.x, pursuer.y - evader.y]) for evader in self.evaders]
            all_pursuers_distance_to_evaders.append(pursuer_distance_to_evaders)
        assert len(all_pursuers_distance_to_evaders) == len(self.pursuers), ("Number of distances should be equal to "
                                                                             "number of pursuers")
        return all_pursuers_distance_to_evaders

    def get_distance_to_other_pursuers(self) -> List[Optional[float]]:
        """
        Get distances between pursuers.

        Returns:
            List[Optional[float]]: List of minimum distances from each pursuer to other pursuers.
        """
        all_pursuers_distance_to_other_pursuers = []

        for pursuer in self.pursuers:
            if pursuer.deactivated:
                all_pursuers_distance_to_other_pursuers.append(None)
                continue

            pursuer_distance_to_other_pursuers = [
                np.linalg.norm([pursuer.x - other_pursuer.x, pursuer.y - other_pursuer.y]) for other_pursuer in
                self.pursuers if other_pursuer is not pursuer]

            all_pursuers_distance_to_other_pursuers.append(np.min(pursuer_distance_to_other_pursuers))

        assert len(all_pursuers_distance_to_other_pursuers) == len(self.pursuers), (
            "Number of distances should be equal "
            "to number of pursuers")
        return all_pursuers_distance_to_other_pursuers

    def update_pursuers_states(self, pursuer_actions: List[int]):
        """
        Update pursuer states.

        Updates pursuer states including position, velocity and trajectory based on actions.

        Args:
            pursuer_actions (List[int]): List of actions for pursuers.
        """
        for i, pursuer_action in enumerate(pursuer_actions):
            pursuer = self.pursuers[i]

            if pursuer.deactivated:
                continue

            pursuer.action_history.append(pursuer_action)

            self.update_rob_state(pursuer, pursuer_action)

            pursuer.trajectory.append([pursuer.x, pursuer.y, pursuer.theta, pursuer.speed, pursuer.velocity[0],
                                       pursuer.velocity[1]])

    def get_distance_to_obstacles(self) -> List[Optional[float]]:
        """
        Get distances from pursuers to obstacles.

        Returns:
            List[Optional[float]]: List of minimum distances from each pursuer to obstacles.
        """
        all_pursuers_distance_to_obstacles = []

        for pursuer in self.pursuers:
            if pursuer.deactivated:
                all_pursuers_distance_to_obstacles.append(None)
                continue

            if not pursuer.perception.observation["statics"]:
                all_pursuers_distance_to_obstacles.append(np.inf)
            else:
                closet_obstacle = pursuer.perception.observation["statics"][0]
                pursuer_distance_to_obstacle = np.linalg.norm([closet_obstacle[0], closet_obstacle[1]])

                all_pursuers_distance_to_obstacles.append(pursuer_distance_to_obstacle)

        assert len(all_pursuers_distance_to_obstacles) == len(self.pursuers), ("Number of distances should be equal to "
                                                                               "number of pursuers")
        return all_pursuers_distance_to_obstacles

    def global_reward(self):
        """
        Calculate global rewards for all pursuers.

        Returns:
            rewards (list): List of reward values for each pursuer.
        """
        rewards = np.zeros(len(self.pursuers))
        num_pursuers_nearby_list = []

        for evader in self.evaders:
            num_pursuers_nearby, related_pursuer_mask = self.count_nearby_pursuers(evader, self.related_distance)
            num_pursuers_nearby_list.append(num_pursuers_nearby)
            if num_pursuers_nearby >= 3:
                reward_factor = max(0, 1.0 - 0.3 * (num_pursuers_nearby - 3))
                rewards += (self.evenly_distributed_reward * reward_factor) * related_pursuer_mask
            elif num_pursuers_nearby >= 5:
                rewards += self.unevenly_distributed_reward * related_pursuer_mask
        if any(num_pursuers_nearby >= 3 for num_pursuers_nearby in num_pursuers_nearby_list) and any(num_pursuers_nearby >= 5 for num_pursuers_nearby in num_pursuers_nearby_list):
            rewards += self.unevenly_distributed_reward * np.ones(len(self.pursuers))

        # Boundary penalty check
        for i, pursuer in enumerate(self.pursuers):
            if pursuer.x < 0 or pursuer.x > self.arena_size or pursuer.y < 0 or pursuer.y > self.arena_size:
                rewards[i] += self.boundary_penalty

        return rewards

    def count_nearby_pursuers(self, evader, distance_threshold):
        """
        Count pursuers within range of an evader.

        Args:
            evader: Evader object with x,y coordinates
            distance_threshold (float): Detection range threshold

        Returns:
            tuple: (count, mask) where count is number of nearby pursuers,
                   mask is boolean array indicating which pursuers are in range
        """
        count = 0
        mask = np.zeros(len(self.pursuers), dtype=bool)
        for i, pursuer in enumerate(self.pursuers):
            distance = np.linalg.norm(np.array([pursuer.x, pursuer.y]) - np.array([evader.x, evader.y]))
            if distance <= distance_threshold:
                count += 1
                mask[i] = True
        return count, mask

    def update_pursuing_status(self):
        """Update pursuers' pursuing status based on distance to nearest evader."""
        for pursuer in self.pursuers:
            nearest_evader_index = self.find_nearest_evader(pursuer)
            distance = np.linalg.norm(
                np.array([pursuer.x, pursuer.y]) - np.array(
                    [self.evaders[nearest_evader_index].x, self.evaders[nearest_evader_index].y])
            )
            pursuer.is_pursuing = distance <= self.related_distance

    def find_nearest_evader(self, pursuer):
        """
        Find index of nearest evader to a pursuer.

        Args:
            pursuer: Pursuer object with x,y coordinates

        Returns:
            int: Index of nearest evader
        """
        distances = [np.linalg.norm(np.array([pursuer.x, pursuer.y]) - np.array([evader.x, evader.y])) for evader in
                     self.evaders]
        return np.argmin(distances)

    def step(self, actions: Tuple[List[int], List[int]]) -> tuple[
        list[Any], ndarray[Any, dtype[floating[_64Bit]]], list[bool], list[dict[str, str]]]:
        """
        Execute one environment simulation step and update the environment.

        Updates environment state based on pursuer and evader actions, calculates rewards,
        termination status, and additional information.

        Args:
            actions (Tuple[List[int], List[int]]): Tuple of pursuer and evader actions.

        Returns:
            Tuple:
                List[Any]: Pursuers' observations list
                List[float]: Reward values list
                List[bool]: Episode termination flags list
                List[dict]: Additional information dictionaries list
        """
        rewards = np.zeros(len(self.pursuers))

        pursuer_actions, evader_actions = actions
        pursuer_actions = utils.to_list(pursuer_actions)
        evader_actions = utils.to_list(evader_actions)

        try:
            assert len(pursuer_actions) == len(
                self.pursuers), "Number of pursuer actions not equal to number of pursuers!"
            assert len(evader_actions) == len(self.evaders), "Number of evader actions not equal to number of evaders!"
        except AssertionError as e:
            logger.error(f"AssertionError: {e}")
            logger.info(
                f"Process {os.getpid()} len(pursuers): {len(self.pursuers)} ,pursuer_actions: {pursuer_actions},"
                f"len(evaders): {len(self.evaders)}, evader_actions: {evader_actions}")
            pdb.set_trace()
            sys.exit(f"Error: Number of actions not equal to number of robots!{e}")

        assert self.check_all_evader_is_captured() is not True, ("All evader-robots are captured, no actions are "
                                                                 "available!")

        # Execute all robot actions

        # Get pre-update pursuer distances to targets
        all_pursuers_dis_before = self.get_pursuers_distance_evaders()

        # Update pursuer states
        self.update_pursuers_states(pursuer_actions)

        # Update evader states
        self.update_evader_state(evader_actions)

        # Get post-update pursuer distances to targets
        all_pursuers_dis_after = self.get_pursuers_distance_evaders()

        all_pursuers_distance_to_other_pursuers = self.get_distance_to_other_pursuers()
        all_purers_distance_to_obstacles = self.get_distance_to_obstacles()

        # Define capture thresholds
        capture_distance = self.pursuers[0].distance_capture  # Capture distance
        safe_distance = 0.5 * capture_distance  # Safety distance

        # Update time penalty and distance rewards
        for i, dis in enumerate(all_pursuers_dis_before):
            if dis is None:
                continue

            # Calculate post-update distance
            min_dis_after = min(all_pursuers_dis_after[i])

            dis_to_pursuer = all_pursuers_distance_to_other_pursuers[i]
            dis_to_obstacle = all_purers_distance_to_obstacles[i]

            # Apply time penalty
            rewards[i] += self.timestep_penalty

            # Apply emergency penalty for unsafe distances
            if min_dis_after < safe_distance or dis_to_pursuer < safe_distance or dis_to_obstacle < safe_distance:
                rewards[i] += self.emergency_penalty

            for dis_to_evader in all_pursuers_dis_after[i]:
                # Fixed reward within capture distance
                if dis_to_evader <= capture_distance:
                    rewards[i] += self.distance_reward
                # Exponential decay reward beyond capture distance
                else:
                    distance_over_saturation = min_dis_after - capture_distance
                    decreasing_reward = 5 * np.exp(self.decay_factor * distance_over_saturation)
                    rewards[i] += decreasing_reward

        global_reward = self.global_reward()
        rewards += global_reward

        # Get pursuer observations
        observations, collisions = self.get_pursuers_observations()

        capture_target_states, capture_angles, num_pursuer_captures, capture_evader_ids = self.get_capture_status_and_info()

        dones = [False] * len(self.pursuers)
        infos = [{"state": "normal"}] * len(self.pursuers)

        # Determine termination conditions
        for idx, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated:
                dones[idx] = True
                if pursuer.collision:
                    infos[idx] = {"state": "deactivated after collision"}
                elif self.check_all_evader_is_captured():
                    infos[idx] = {"state": "deactivated after all evaders were captured"}
                else:
                    logger.info(f"Process {os.getpid()}" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    logger.error(f"wrong deactivated🥲.")
                    raise RuntimeError("Robot being deactivated can only be caused by collision or all evaders being "
                                       "captured!")

            if self.episode_time_steps >= self.episode_max_length:
                dones[idx] = True
                infos[idx] = {"state": "too long episode"}
            elif collisions[idx]:
                rewards[idx] += self.collision_penalty
                dones[idx] = True
                infos[idx] = {"state": "collision"}
            elif capture_target_states[idx]:
                rewards[idx] += self.goal_reward * self.compute_capture_reward_factor(
                    capture_angles=capture_angles[idx],
                    pursuer_count=num_pursuer_captures[idx])
                infos[idx] = {
                    "state": f"✌️capture one evader id-{pursuer.captured_evaderId_list[-1]}, total captured evader-{pursuer.captured_evaderId_list} "}
                pursuer.is_current_target_captured = False
            else:
                dones[idx] = False
                infos[idx] = {"state": "normal"}

        self.episode_time_steps += 1
        self.total_time_steps += 1

        return observations, rewards, dones, infos

    def get_evenly_distributed_reward(self, angles: List[float]) -> float:
        """
        Calculate evenly distributed reward based on angle distribution.

        Args:
            angles (List[float]): List of angles between evaders

        Returns:
            float: Calculated distribution reward
        """
        if not angles:
            return 0.0

        # Calculate angle statistics
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)

        # Compute distribution reward
        reward = (2 * np.pi) / len(angles) * np.exp(-std_angle) - abs(mean_angle - 2 * np.pi / 3)

        return reward

    def update_rob_state(self, rob: Pursuer | Evader, rob_action: int):
        """
        Update robot state through multiple steps.

        Args:
            rob (Pursuer | Evader): Robot instance to update
            rob_action (int): Action index to execute
        """
        for _ in range(rob.N):
            current_velocity = self.get_current_velocity(rob.x, rob.y)
            rob.update_state(rob_action, current_velocity)

    def compute_capture_reward_factor(self, capture_angles: List[float], pursuer_count: int) -> float:
        """
        Calculate capture reward factor based on formation quality.

        Args:
            capture_angles (List[float]): Angles between capturing pursuers
            pursuer_count (int): Number of participating pursuers

        Returns:
            float: Capture reward multiplier
        """
        if not capture_angles or pursuer_count <= 0:
            return 0.0

        # Calculate angle distribution quality
        sigma = np.std(capture_angles)

        # Compute formation quality factor
        capture_reward_factor = ((2 * np.pi) / pursuer_count) * np.exp(-sigma)

        return capture_reward_factor

    def out_of_boundary(self) -> bool:
        """
        Check if any robot exceeds map boundaries.

        Returns:
            bool: True if any robot is out of bounds, False otherwise
        """
        x_out = any(rob.x < 0.0 or rob.x > self.width for rob in self.pursuers + self.evaders)
        y_out = any(rob.y < 0.0 or rob.y > self.height for rob in self.pursuers + self.evaders)
        return x_out or y_out

    def get_pursuers_observations(self) -> Tuple[List[Any], List[bool]]:
        """
        Retrieve observations for all pursuers.

        Returns:
            tuple:
                List[Any]: Pursuers' observation data
                List[bool]: Collision status list
        """
        observations = []
        collisions = []

        for pursuer in self.pursuers:
            observation, collision = pursuer.perception_output(self.obstacles,
                                                               self.pursuers,
                                                               self.evaders,
                                                               self.observation_in_robot_frame)
            observations.append(observation)
            collisions.append(collision)

        return observations, collisions

    def get_evaders_observation(self) -> Tuple[List[Any], List[bool]]:
        """
        Retrieve observations for all evaders.

        Returns:
            tuple:
                List[Any]: Evaders' observation data
                List[bool]: Collision status list
        """
        observations = []
        collisions = []

        for evader in self.evaders:
            observation, collision = evader.perception_output(obstacles=self.obstacles,
                                                              pursuers=self.pursuers,
                                                              evaders=self.evaders,
                                                              in_robot_frame=self.observation_in_robot_frame)
            observations.append(observation)
            collisions.append(collision)
        return observations, collisions

    def get_pursuers_captured_state(self) -> List[bool]:
        """
        Retrieve pursuers' capture status.

        Returns:
            List[bool]: Capture success status for each pursuer
        """
        # Reset temporary states
        for pursuer in self.pursuers:
            pursuer.is_current_target_captured = False
        for evader in self.evaders:
            evader.deactivated = False  # Temporary state reset

        # Check capture status
        capture_targets = []
        for pursuer in self.pursuers:
            pursuer.check_capture_current_target(pursuers=self.pursuers, evaders=self.evaders)
            capture_targets.append(pursuer.is_current_target_captured)

        # Restore permanent deactivation states
        for pursuer in self.pursuers:
            for evader in self.evaders:
                if evader.deactivated:
                    continue
                if evader.id in pursuer.captured_evaderId_list:
                    evader.deactivated = True

        return capture_targets

    def get_capture_status_and_info(self) -> Tuple[List[bool], List[List[float]], List[int], List[Optional[int]]]:
        """
        Retrieve detailed capture information.

        Returns:
            tuple:
                List[bool]: Capture success status
                List[List[float]]: Capture angle distributions
                List[int]: Number of participating pursuers
                List[Optional[int]]: Captured evader IDs
        """
        capture_targets = []
        capture_angles = []
        num_pursuer_captures = []
        capture_evader_ids = []

        for pursuer in self.pursuers:
            is_captured, capture_angle, num_pursuer_capture, capture_evader_id = (
                pursuer.check_capture_current_target(pursuers=self.pursuers, evaders=self.evaders))

            capture_targets.append(pursuer.is_current_target_captured)
            capture_angles.append(capture_angle)
            num_pursuer_captures.append(num_pursuer_capture)
            capture_evader_ids.append(capture_evader_id)

        # Restore permanent evader states
        for pursuer in self.pursuers:
            for evader in self.evaders:
                if evader.id in pursuer.captured_evaderId_list:
                    evader.deactivated = True

        return capture_targets, capture_angles, num_pursuer_captures, capture_evader_ids

    def get_capture_info(self) -> Tuple[List[List[float]], List[int], List[Optional[int]]]:
        """
        Retrieve capture metadata.

        Returns:
            tuple:
                List[List[float]]: Angle distributions during capture
                List[int]: Number of participating pursuers
                List[Optional[int]]: Captured evader IDs
        """
        capture_angles = []
        num_pursuer_captures = []
        capture_evader_ids = []

        for pursuer in self.pursuers:
            _, capture_angle, num_pursuer_capture, capture_evader_id = (
                pursuer.check_capture_current_target(pursuers=self.pursuers, evaders=self.evaders))

            capture_angles.append(capture_angle)
            num_pursuer_captures.append(num_pursuer_capture)
            capture_evader_ids.append(capture_evader_id)

        # Restore evader states
        for pursuer in self.pursuers:
            for evader in self.evaders:
                if evader.deactivated:
                    continue
                if evader.id in pursuer.captured_evaderId_list:
                    evader.deactivated = True

        return capture_angles, num_pursuer_captures, capture_evader_ids

    def check_pursuer_start_position(self, pursuer_start: np.ndarray) -> bool:
        """
        Validate pursuer start position.

        Args:
            pursuer_start (np.ndarray): Proposed start coordinates [x, y]

        Returns:
            bool: True if position meets safety requirements
        """
        # Check distance to other pursuers
        for pursuer in self.pursuers:
            if np.linalg.norm(pursuer_start - pursuer.start) <= self.clear_r:
                return False

        # Check distance to evaders
        for evader in self.evaders:
            if np.linalg.norm(pursuer_start - evader.start) <= self.min_pursuer_evader_init_dis:
                return False

        return True

    def check_evader_start_position(self, evader_start: np.ndarray) -> bool:
        """
        Validate evader start position.

        Args:
            evader_start (np.ndarray): Proposed start coordinates [x, y]

        Returns:
            bool: True if position meets safety requirements
        """
        # Check distance to pursuers
        for pursuer in self.pursuers:
            if np.linalg.norm(pursuer.start - evader_start) <= self.min_pursuer_evader_init_dis:
                return False

        # Check distance to other evaders
        for evader in self.evaders:
            if np.linalg.norm(evader_start - evader.start) <= self.clear_r:
                return False

        return True

    def check_core(self, core_j: Core) -> bool:
        """
        Validate vortex core position.

        Ensures the vortex core is within map boundaries and doesn't overlap with:
        - Other vortex cores
        - Obstacles
        - Robot starting positions

        Args:
            core_j (Core): Vortex core instance to validate

        Returns:
            bool: True if position is valid, False otherwise
        """
        # Check map boundaries
        if core_j.x - self.vortex_core_radius < 0.0 or core_j.x + self.vortex_core_radius > self.width:
            return False
        if core_j.y - self.vortex_core_radius < 0.0 or core_j.y + self.vortex_core_radius > self.height:
            return False

        # Check distance to robot start positions
        for rob in self.pursuers + self.evaders:
            core_pos = np.array([core_j.x, core_j.y])
            dis_s = core_pos - rob.start
            if np.linalg.norm(dis_s) < self.vortex_core_radius + self.clear_r:
                return False

        # Check interactions with other vortex cores
        for core_i in self.cores:
            dx = core_i.x - core_j.x
            dy = core_i.y - core_j.y
            dis = np.sqrt(dx * dx + dy * dy)

            if core_i.clockwise == core_j.clockwise:
                # Same rotation direction - validate boundary speeds
                boundary_i = core_i.Gamma / (2 * np.pi * self.v_rel_max)
                boundary_j = core_j.Gamma / (2 * np.pi * self.v_rel_max)
                if dis < boundary_i + boundary_j:
                    return False
            else:
                # Opposite rotation - validate velocity ratios
                gamma_l = max(core_i.Gamma, core_j.Gamma)
                gamma_s = min(core_i.Gamma, core_j.Gamma)
                v_1 = gamma_l / (2 * np.pi * (dis - 2 * self.vortex_core_radius))
                v_2 = gamma_s / (2 * np.pi * self.vortex_core_radius)
                if v_1 > self.p * v_2:
                    return False

        return True

    def check_obstacle(self, obs: Obstacle) -> bool:
        """
        Validate obstacle position.

        Ensures the obstacle is within map boundaries and doesn't overlap with:
        - Other obstacles
        - Vortex cores
        - Robot starting positions

        Args:
            obs (Obstacle): Obstacle instance to validate

        Returns:
            bool: True if position is valid, False otherwise
        """
        # Check map boundaries
        if obs.x - obs.r < 0.0 or obs.x + obs.r > self.width:
            return False
        if obs.y - obs.r < 0.0 or obs.y + obs.r > self.height:
            return False

        # Check distance to robot start positions
        for rob in self.pursuers + self.evaders:
            obs_pos = np.array([obs.x, obs.y])
            dis_s = obs_pos - rob.start
            if np.linalg.norm(dis_s) < obs.r + self.clear_r:
                return False

        min_obstructions_dis = 5  # Minimum separation distance

        # Check distance to vortex cores
        for core in self.cores:
            dx = core.x - obs.x
            dy = core.y - obs.y
            dis = np.sqrt(dx * dx + dy * dy)
            if dis <= self.vortex_core_radius + obs.r + min_obstructions_dis:
                return False

        # Check distance to other obstacles
        for obstacle in self.obstacles:
            dx = obstacle.x - obs.x
            dy = obstacle.y - obs.y
            dis = np.sqrt(dx * dx + dy * dy)
            if dis <= obstacle.r + obs.r + min_obstructions_dis:
                return False

        return True

    def get_current_velocity(self, x: float, y: float) -> np.ndarray:
        """
        Calculate ocean current velocity at specified coordinates.

        Computes cumulative velocity from all vortex cores considering:
        - Distance from each core
        - Rotation direction (clockwise/counter-clockwise)
        - Circulation strength (Gamma)

        Args:
            x (float): X-coordinate
            y (float): Y-coordinate

        Returns:
            np.ndarray: Current velocity vector [vx, vy]
        """
        if len(self.cores) == 0:
            return np.zeros(2)

        # Query nearest vortex cores
        d, idx = self.core_centers.query(np.array([x, y]), k=len(self.cores))
        if isinstance(idx, np.int64):
            idx = [idx]

        v_radial_set = []
        v_velocity = np.zeros((2, 1))
        for i in list(idx):
            core = self.cores[i]
            v_radial = np.array([[core.x - x], [core.y - y]])

            # Avoid duplicate contributions
            for v in v_radial_set:
                project = np.transpose(v) @ v_radial
                if project[0, 0] > 0:
                    continue

            v_radial_set.append(v_radial)
            dis = np.linalg.norm(v_radial)
            v_radial /= dis
            if core.clockwise:
                rotation = np.array([[0., -1.], [1., 0.]])
            else:
                rotation = np.array([[0., 1.], [-1., 0.]])
            v_tangent = rotation @ v_radial
            speed = self.compute_speed(core.Gamma, float(dis))
            v_velocity += v_tangent * speed

        return np.array([v_velocity[0, 0], v_velocity[1, 0]])

    def compute_speed(self, gamma: float, d: float) -> float:
        """
        Calculate tangential velocity at distance d from vortex core.

        Velocity profile:
        - Linear increase within core radius
        - Inverse relationship beyond core radius

        Args:
            gamma (float): Circulation strength
            d (float): Distance from core center

        Returns:
            float: Tangential velocity magnitude
        """
        if d <= self.vortex_core_radius:
            return gamma / (2 * np.pi * self.vortex_core_radius * self.vortex_core_radius) * d
        else:
            return gamma / (2 * np.pi * d)

    def reset_with_eval_config(self, eval_config: dict):
        """
        Reset environment with evaluation configuration.

        Loads predefined environment parameters including:
        - Map dimensions
        - Vortex core properties
        - Obstacle placements
        - Robot initial states

        Args:
            eval_config (dict): Configuration dictionary containing:
                - env: Environment parameters
                - cores: Vortex core specifications
                - obstacles: Obstacle positions
                - pursuers/evaders: Robot configurations
        """
        self.episode_time_steps = 0

        # Load environment parameters
        env_config = eval_config["env"]
        self.sd = env_config["seed"]
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.vortex_core_radius = env_config["vortex_core_radius"]
        self.v_rel_max = env_config["v_rel_max"]
        self.p = env_config["p"]
        self.v_range = copy.deepcopy(env_config["v_range"])
        self.obs_r_range = copy.deepcopy(env_config["obs_r_range"])
        self.clear_r = env_config["clear_r"]
        self.min_pursuer_evader_init_dis = env_config["min_pursuer_evader_init_dis"]
        self.timestep_penalty = env_config["timestep_penalty"]
        self.collision_penalty = env_config["collision_penalty"]
        self.goal_reward = env_config["goal_reward"]

        # Initialize vortex cores
        self.cores.clear()
        centers = None
        for i in range(len(env_config["cores"]["positions"])):
            center = env_config["cores"]["positions"][i]
            clockwise = env_config["cores"]["clockwise"][i]
            gamma = env_config["cores"]["Gamma"][i]
            core = Core(center[0], center[1], clockwise, gamma)
            self.cores.append(core)
            if centers is None:
                centers = np.array([[core.x, core.y]])
            else:
                c = np.array([[core.x, core.y]])
                centers = np.vstack((centers, c))

        if centers is not None:
            self.core_centers = KDTree(centers)

        # Initialize obstacles
        self.obstacles.clear()
        for i in range(len(env_config["obstacles"]["positions"])):
            center = env_config["obstacles"]["positions"][i]
            r = env_config["obstacles"]["r"][i]
            obs = Obstacle(center[0], center[1], r)
            self.obstacles.append(obs)

        # Configure pursuers
        self.pursuers.clear()
        for i in eval_config["pursuers"]["index"]:
            pursuer = Pursuer(i)
            # Set configuration parameters
            pursuer.robot_type = 'pursuer'
            pursuer.dt = eval_config["pursuers"]["dt"][i]
            pursuer.N = eval_config["pursuers"]["N"][i]
            pursuer.length = eval_config["pursuers"]["length"][i]
            pursuer.width = eval_config["pursuers"]["width"][i]
            pursuer.obs_dis = eval_config["pursuers"]["obs_dis"][i]
            pursuer.r = eval_config["pursuers"]["r"][i]
            pursuer.detect_r = eval_config["pursuers"]["detect_r"][i]
            pursuer.a = np.array(eval_config["pursuers"]["a"][i])
            pursuer.w = np.array(eval_config["pursuers"]["w"][i])
            pursuer.perception.range = eval_config["pursuers"]["perception"]["range"][i]
            pursuer.perception.angle = eval_config["pursuers"]["perception"]["angle"][i]
            pursuer.max_speed = eval_config["pursuers"]["max_speed"][i]
            pursuer.start = tuple(eval_config["pursuers"]["start"][i])
            pursuer.init_theta = eval_config["pursuers"]["init_theta"][i]
            pursuer.init_speed = eval_config["pursuers"]["init_speed"][i]
            pursuer.distance_capture = eval_config["pursuers"]["distance_capture"][i]
            pursuer.angle_capture = eval_config["pursuers"]["angle_capture"][i]
            # Initialize dynamics
            pursuer.compute_k()
            pursuer.compute_actions()
            current_v = self.get_current_velocity(float(pursuer.start[0]), float(pursuer.start[1]))
            pursuer.reset_state(current_velocity=current_v)
            self.pursuers.append(pursuer)

        # Configure evaders
        self.evaders.clear()
        for i in eval_config["evaders"]["index"]:
            evader = Evader(i)
            # Set configuration parameters
            evader.robot_type = 'evader'
            evader.dt = eval_config["evaders"]["dt"][i]
            evader.N = eval_config["evaders"]["N"][i]
            evader.length = eval_config["evaders"]["length"][i]
            evader.width = eval_config["evaders"]["width"][i]
            evader.obs_dis = eval_config["evaders"]["obs_dis"][i]
            evader.r = eval_config["evaders"]["r"][i]
            evader.detect_r = eval_config["evaders"]["detect_r"][i]
            evader.a = np.array(eval_config["evaders"]["a"][i])
            evader.w = np.array(eval_config["evaders"]["w"][i])
            evader.perception.range = eval_config["evaders"]["perception"]["range"][i]
            evader.perception.angle = eval_config["evaders"]["perception"]["angle"][i]
            evader.max_speed = eval_config["evaders"]["max_speed"][i]
            evader.start = tuple(eval_config["evaders"]["start"][i])
            evader.init_theta = eval_config["evaders"]["init_theta"][i]
            evader.init_speed = eval_config["evaders"]["init_speed"][i]
            # Initialize dynamics
            evader.compute_k()
            evader.compute_actions()
            current_v = self.get_current_velocity(float(evader.start[0]), float(evader.start[1]))
            evader.reset_state(current_velocity=current_v)
            self.evaders.append(evader)

        return self.get_pursuers_observations()

    def episode_data(self) -> dict:
        """
        Serialize environment state for persistence.

        Captures complete environment and robot state including:
        - Environment parameters
        - Vortex core configurations
        - Obstacle positions
        - Robot parameters and trajectories

        Returns:
            dict: Structured dictionary containing:
                - env: Environment parameters
                - cores: Vortex core data
                - obstacles: Obstacle data
                - pursuers/evaders: Robot states
        """
        episode = dict()

        # Environment parameters
        episode["env"] = {}
        episode["env"]["seed"] = self.sd
        episode["env"]["width"] = self.width
        episode["env"]["height"] = self.height
        episode["env"]["vortex_core_radius"] = self.vortex_core_radius
        episode["env"]["v_rel_max"] = self.v_rel_max
        episode["env"]["p"] = self.p
        episode["env"]["v_range"] = copy.deepcopy(self.v_range)
        episode["env"]["obs_r_range"] = copy.deepcopy(self.obs_r_range)
        episode["env"]["clear_r"] = self.clear_r
        episode["env"]["min_pursuer_evader_init_dis"] = self.min_pursuer_evader_init_dis
        episode["env"]["timestep_penalty"] = self.timestep_penalty
        episode["env"]["collision_penalty"] = self.collision_penalty
        episode["env"]["goal_reward"] = self.goal_reward

        # Vortex core data
        episode["env"]["cores"] = {}
        episode["env"]["cores"]["positions"] = []
        episode["env"]["cores"]["clockwise"] = []
        episode["env"]["cores"]["Gamma"] = []
        for core in self.cores:
            episode["env"]["cores"]["positions"].append([core.x, core.y])
            episode["env"]["cores"]["clockwise"].append(core.clockwise)
            episode["env"]["cores"]["Gamma"].append(core.Gamma)

        # Obstacle data
        episode["env"]["obstacles"] = {}
        episode["env"]["obstacles"]["positions"] = []
        episode["env"]["obstacles"]["r"] = []
        for obs in self.obstacles:
            episode["env"]["obstacles"]["positions"].append([obs.x, obs.y])
            episode["env"]["obstacles"]["r"].append(obs.r)

        # Pursuer configurations
        episode["pursuers"] = {}
        episode["pursuers"]["index"] = []
        episode["pursuers"]["robot_type"] = []
        episode["pursuers"]["dt"] = []
        episode["pursuers"]["N"] = []
        episode["pursuers"]["length"] = []
        episode["pursuers"]["width"] = []
        episode["pursuers"]["obs_dis"] = []
        episode["pursuers"]["r"] = []
        episode["pursuers"]["detect_r"] = []
        episode["pursuers"]["a"] = []
        episode["pursuers"]["w"] = []
        episode["pursuers"]["perception"] = {}
        episode["pursuers"]["perception"]["range"] = []
        episode["pursuers"]["perception"]["angle"] = []
        episode["pursuers"]["perception"]["communication_range"] = []
        episode["pursuers"]["perception"]["communication_angle"] = []
        episode["pursuers"]["max_speed"] = []
        episode["pursuers"]["start"] = []
        episode["pursuers"]["init_theta"] = []
        episode["pursuers"]["init_speed"] = []
        episode["pursuers"]["action_history"] = []
        episode["pursuers"]["trajectory"] = []
        episode["pursuers"]["target_id"] = []
        episode["pursuers"]["target_position"] = []
        episode["pursuers"]["distance_capture"] = []
        episode["pursuers"]["angle_capture"] = []

        for pursuer in self.pursuers:
            episode["pursuers"]["index"].append(pursuer.id)
            episode["pursuers"]["robot_type"].append(pursuer.robot_type)
            episode["pursuers"]["dt"].append(pursuer.dt)
            episode["pursuers"]["N"].append(pursuer.N)
            episode["pursuers"]["length"].append(pursuer.length)
            episode["pursuers"]["width"].append(pursuer.width)
            episode["pursuers"]["obs_dis"].append(pursuer.obs_dis)
            episode["pursuers"]["r"].append(pursuer.r)
            episode["pursuers"]["detect_r"].append(pursuer.detect_r)
            episode["pursuers"]["a"].append(list(pursuer.a))
            episode["pursuers"]["w"].append(list(pursuer.w))
            episode["pursuers"]["perception"]["range"].append(pursuer.perception.range)
            episode["pursuers"]["perception"]["angle"].append(pursuer.perception.angle)
            episode["pursuers"]["max_speed"].append(pursuer.max_speed)
            episode["pursuers"]["start"].append(list(pursuer.start))
            episode["pursuers"]["init_theta"].append(pursuer.init_theta)
            episode["pursuers"]["init_speed"].append(pursuer.init_speed)
            episode["pursuers"]["action_history"].append(copy.deepcopy(pursuer.action_history))
            episode["pursuers"]["trajectory"].append(copy.deepcopy(pursuer.trajectory))
            episode["pursuers"]["distance_capture"].append(pursuer.distance_capture)
            episode["pursuers"]["angle_capture"].append(pursuer.angle_capture)

        # Evader configurations
        episode["evaders"] = {}
        episode["evaders"]["index"] = []
        episode["evaders"]["robot_type"] = []
        episode["evaders"]["dt"] = []
        episode["evaders"]["N"] = []
        episode["evaders"]["length"] = []
        episode["evaders"]["width"] = []
        episode["evaders"]["obs_dis"] = []
        episode["evaders"]["r"] = []
        episode["evaders"]["detect_r"] = []
        episode["evaders"]["a"] = []
        episode["evaders"]["w"] = []
        episode["evaders"]["perception"] = {}
        episode["evaders"]["perception"]["range"] = []
        episode["evaders"]["perception"]["angle"] = []
        episode["evaders"]["max_speed"] = []
        episode["evaders"]["start"] = []
        episode["evaders"]["init_theta"] = []
        episode["evaders"]["init_speed"] = []
        episode["evaders"]["action_history"] = []
        episode["evaders"]["trajectory"] = []

        for evader in self.evaders:
            episode["evaders"]["index"].append(evader.id)
            episode["evaders"]["robot_type"].append(evader.robot_type)
            episode["evaders"]["dt"].append(evader.dt)
            episode["evaders"]["N"].append(evader.N)
            episode["evaders"]["length"].append(evader.length)
            episode["evaders"]["width"].append(evader.width)
            episode["evaders"]["obs_dis"].append(evader.obs_dis)
            episode["evaders"]["r"].append(evader.r)
            episode["evaders"]["detect_r"].append(evader.detect_r)
            episode["evaders"]["a"].append(list(evader.a))
            episode["evaders"]["w"].append(list(evader.w))
            episode["evaders"]["perception"]["range"].append(evader.perception.range)
            episode["evaders"]["perception"]["angle"].append(evader.perception.angle)
            episode["evaders"]["max_speed"].append(evader.max_speed)
            episode["evaders"]["start"].append(list(evader.start))
            episode["evaders"]["init_theta"].append(evader.init_theta)
            episode["evaders"]["init_speed"].append(evader.init_speed)
            episode["evaders"]["action_history"].append(copy.deepcopy(evader.action_history))
            episode["evaders"]["trajectory"].append(copy.deepcopy(evader.trajectory))

        return episode

    def save_episode(self, filename: str):
        """
        Save current episode data to file.

        Serializes and persists the complete environment state including:
        - Environment configuration
        - Vortex core parameters
        - Obstacle positions
        - Robot states and trajectories

        Args:
            filename (str): Output file path for saving JSON data
        """
        episode = self.episode_data()
        with open(filename, "w") as file:
            json.dump(episode, file)
