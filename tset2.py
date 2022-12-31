import carla
import abc
import glob
import os
import sys
from types import LambdaType
from collections import deque
from collections import namedtuple
# from srunner.challenge.utils.route_manipulation import interpolate_trajectory


sys.path.insert(0, '/home/haojiachen/CARLA_0.9.13/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

import random
import time
import numpy as np

print('connecting to Carla server...')
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
print('Carla server connected!')
world = client.load_world('Town04')
# settings = world.get_settings()
# settings.fixed_delta_seconds = 0.1
# settings.synchronous_mode = True
# world.apply_settings(settings)

waypoints = world.get_map().generate_waypoints(1)
# for w in waypoints:
#     world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
#                             color=carla.Color(r=255, g=0, b=0), life_time=120.0,
#                             persistent_lines=True)
# while True:
#     world.wait_for_tick()

blueprint_library = world.get_blueprint_library()
ego_bp = random.choice(world.get_blueprint_library().filter("vehicle.lincoln*"))
ego_bp.set_attribute('color', "255,0,0")

ego_bp2 = random.choice(world.get_blueprint_library().filter("vehicle.lincoln*"))
ego_bp2.set_attribute('color', "255,128,0")

# target_transform = world.get_map().get_spawn_points()[10]
# # destination = [target_transform.location.x,target_transform.location.y,target_transform.location.z]
# print(target_transform)
map = world.get_map()
# waypoints = map.generate_waypoints(396)
# print(waypoints)

# target_transform = world.get_map().get_spawn_points()[110]
# Get waypoint


actor_list = []
transform1 = carla.Transform(
    carla.Location(x=255.3, y=-271.3, z=0.275307),
    carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
# random.choice(self.world.get_map().get_spawn_points())
vehicle1 = world.spawn_actor(ego_bp, transform1)

waypoint_ego = vehicle1.get_transform()
waypoint1 = map.get_waypoint(carla.Location(x=255.3, y=-271.3, z=0.275307))
world.debug.draw_string(waypoint1.transform.location, 'O', draw_shadow=False,
                        color=carla.Color(r=255, g=255, b=0), life_time=120.0,
                        persistent_lines=True)
next_waypoints = []
for i in range(1, 50):
    next_waypoint = waypoint1.next(i)
    for w in next_waypoint:
        if w.transform.location.x >= waypoint1.transform.location.x-0.045:
            next_waypoints.append(w)
# next_waypoint = waypoint1.next(30)
for w in next_waypoints:
    world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                            color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                            persistent_lines=True)
world.wait_for_tick()
world.tick()
# location = vehicle1.get_location()

# waypoint2 = map.get_waypoint(carla.Location(x=-6.1, y=117.1, z=0.275307))
transform2 = carla.Transform(carla.Location(x=2.0, y=147.3, z=0.275307),
                             carla.Rotation(pitch=0.000000, yaw=270, roll=0.000000))
# random.choice(self.world.get_map().get_spawn_points())
vehicle2 = world.spawn_actor(ego_bp2, transform2)

actor_list.append(vehicle1)
actor_list.append(vehicle2)

agent1 = BasicAgent(vehicle1)
agent2 = BasicAgent(vehicle2)
# location1 = agent1.vehicle.get_location()

target1 = map.get_waypoint(carla.Location(x=37.2, y=134.0, z=0.275307))
destination1 = [target1.transform.location.x, target1.transform.location.y, target1.transform.location.z]
agent2.set_destination(carla.Location(x=37.2, y=134.0, z=0.275307))

target2 = map.get_waypoint(carla.Location(x=1.7, y=105.1, z=0.275307))
destination2 = [target2.transform.location.x, target2.transform.location.y, target2.transform.location.z]
agent2.set_destination(carla.Location(x=1.7, y=105.1, z=0.275307))

while True:
    if agent1.done():
        print("The target has been reached, stopping the simulation")
        break
    control1 = agent1.run_step()
    vehicle1.apply_control(control1)
    control2 = agent2.run_step()
    vehicle2.apply_control(control2)

for agent in actor_list:
    agent.destroy()
print('done')
