#!/usr/bin/env python

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla


class CarlaEnv11(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        # parameters

        self.dt = params['dt']
        self.max_time_episode = params['max_time_episode']
        self.desired_speed = params['desired_speed']
        self.dests = [[145, 58.910496, 0.275307]]

        # Connect to carla server and get world object
        print('connecting to Carla server...')
        client = carla.Client('localhost', params['port'])
        client.set_timeout(10.0)
        self.world = client.load_world(params['town'])
        print('Carla server connected!')

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # # Get spawn points
        # self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # self.walker_spawn_points = []

        # Create the ego vehicle blueprint
        self.ego_bp = random.choice(self.world.get_blueprint_library().filter("vehicle.lincoln*"))
        self.ego_bp.set_attribute('color', "255,0,0")
        self.surround_bp = random.choice(self.world.get_blueprint_library().filter("vehicle.carlamotors*"))
        self.surround_bp.set_attribute('color', "255,128,0")

        # # Camera sensor
        self.camera_img = np.zeros((384, 216, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(384))
        self.camera_bp.set_attribute('image_size_y', str(216))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # self.camera_img2 = np.zeros((1920, 1080, 3), dtype=np.uint8)
        # self.camera_trans2 = carla.Transform(carla.Location(x=0.8, z=1.7))
        # self.camera_bp2 = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # # Modify the attributes of the blueprint to set image resolution and field of view.
        # self.camera_bp2.set_attribute('image_size_x', str(1920))
        # self.camera_bp2.set_attribute('image_size_y', str(1080))
        # self.camera_bp2.set_attribute('fov', '110')
        # # Set the time in seconds between sensor captures
        # self.camera_bp2.set_attribute('sensor_tick', '0.02')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # # Initialize the renderer
        # self._init_renderer()

        # # Get pixel grid points
        # if self.pixor:
        #   x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
        #   x, y = x.flatten(), y.flatten()
        #   self.pixel_grid = np.vstack((x, y)).T

    def reset(self):
        # Clear sensor objects
        self.collision = False
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        # self.camera_sensor2 = None

        self.location_flag = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb',
                                'sensor.camera.semantic_segmentation', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

        # Disable sync mode
        self._set_synchronous_mode(False)
        spaw_points = self.world.get_map().get_spawn_points()

        # self.vehicle_spawn_points0 = carla.Transform(
        #     carla.Location(x=181.899918 + np.random.uniform(-10, 10), y=58.910496, z=0.275307),
        #     carla.Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000))
        self.vehicle_spawn_points0 = carla.Transform(
            carla.Location(x=181.5, y=58.910496, z=0.275307),
            carla.Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000))
        self.vehicle_spawn_points1 = carla.Transform(carla.Location(x=157.899918, y=54.910496, z=0.275307),
                                                     carla.Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000))
        self.ego = self.world.spawn_actor(self.ego_bp, self.vehicle_spawn_points0)

        self.surround = self.world.try_spawn_actor(self.surround_bp, self.vehicle_spawn_points1)
        self.surround.set_autopilot(False)

        self.ego.set_destination(agent.vehicle.get_location(), )

        # spawing a walker
        blueprint = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        spawn_point = carla.Transform(carla.Location(x=151.5, y=53, z=0.275307),
                                      carla.Rotation(pitch=0.000000, yaw=90.852554, roll=0.000000))
        # spawn_points = self.world.get_map().get_spawn_points()
        # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.person = self.world.spawn_actor(blueprint, spawn_point)

        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), self.person)
        # start walker
        walker_controller_actor.start()
        # set walk to random point
        # walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
        walker_controller_actor.go_to_location(carla.Location(x=151.5, y=68, z=0.275307))
        # random max speed
        walker_controller_actor.set_max_speed(1.5)  # max speed between 1 and 2 (default is 1.4 m/s)

        self.ego.set_target_velocity(carla.Vector3D(-self.desired_speed, 0, 0))

        # # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            image = np.reshape(array, (data.height, data.width, 4))

            # Get the r channel
            sem = image[:, :, 2]
            # print(sem)
            m = len(sem[0, :])
            if self.location_flag == None:
                for i in range(len(sem[:, 0])):
                    for j in range(int(m / 2)):
                        if sem[i][j + int(m / 2)] == 4:
                            self.location_flag = True

            # print(self.location_flag)

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        # Set ego information for render

        return self._get_obs()

    def step(self, action):
        # Calculate acceleration and steering
        throttle = 0
        brake = 0

        if action == 0:
            brake = 0.5
        elif action == 1:
            brake = 0.25
        elif action == 2:
            throttle = 0.2
        elif action == 3:
            throttle = 0.4
        elif action == 4:
            throttle = 0.6
        else:
            throttle = 0.8

        steer = 0
        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)
        # print(self.ego.get_velocity().x)
        self.world.tick()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        self.info = None

        return (self._get_obs(), self._get_reward(), self._terminal(), self.info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _get_obs(self):
        """Get the observations."""

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y

        surround_x = self.surround.get_transform().location.x
        surround_y = self.surround.get_transform().location.y

        person_x = 150
        person_y = 50
        if self.location_flag:
            person_x = self.person.get_transform().location.x
            person_y = self.person.get_transform().location.y
        person_v = self.person.get_velocity()
        egovehicle_v = self.ego.get_velocity()
        print(egovehicle_v)
        obs = [surround_x - ego_x, surround_y - ego_y, person_x - ego_x, person_y - ego_y, person_v.y,
               egovehicle_v.x]  # relative location

        return obs

    def _get_reward(self):
        """Calculate the step reward."""
        # # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        r_speed = speed
        if speed > self.desired_speed:
            r_speed = speed - (speed - self.desired_speed) ** 2
        #
        # reward for collision
        r_collision = 0
        ego_x = self.ego.get_transform().location.x
        ego_y = self.ego.get_transform().location.y

        person_x = self.person.get_transform().location.x
        person_y = self.person.get_transform().location.y
        if abs(person_x - ego_x) < 3 and abs(person_y - ego_y) < 2.5:
            r_collision = -1

        r_time = 0
        if self.time_step > self.max_time_episode:
            r_time = -1

        # cost for cceleration
        a = self.ego.get_acceleration()
        acc = np.sqrt(a.x ** 2 + a.y ** 2)
        r_acc = -abs(acc ** 2)

        ego_x = self.ego.get_transform().location.x
        ego_y = self.ego.get_transform().location.y

        r_success = 0
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 2:
                    r_success = 1

        r = 1000 * r_collision + r_speed + r_acc + 500 * r_success + 200 * r_time
        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # # Get ego state
        # ego_x, ego_y = get_pos(self.ego)
        #
        # # If collides
        # if len(self.collision_hist)>0:
        #   return True
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)

        ego_x = self.ego.get_transform().location.x
        ego_y = self.ego.get_transform().location.y

        person_x = self.person.get_transform().location.x
        person_y = self.person.get_transform().location.y
        if abs(person_x - ego_x) < 3 and abs(person_y - ego_y) < 2.5:
            print(abs(person_x - ego_x), abs(person_y - ego_y))
            print("ego vehicle speed:", speed)
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 2:
                    return True

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()
