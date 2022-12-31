# 对carla的接口进行简单的介绍
import carla
import random
import numpy as np

# 创建client
client = carla.Client('localhost', 2000)
# 设置连接超时时间
client.set_timeout(10.0)
# 选择world
world = client.load_world('Town03')
# 设置天气
world.set_weather(carla.WeatherParameters.ClearNoon)
###################
# 设置仿真时间与真实时间一致，间隔为0.1s
settings = world.get_settings()
settings.fixed_delta_seconds = 0.1
settings.synchronous_mode = True
world.apply_settings(settings)
##################
# 生成汽车
# 随机生成一个汽车的蓝图（就是选车型）
ego_bp = random.choice(world.get_blueprint_library().filter("vehicle.lincoln*"))
# 设置车辆颜色
ego_bp.set_attribute('color', "255,0,0")
# 设置初始位置点
vehicle_spawn_points0 = carla.Transform(
    carla.Location(x=181.5, y=58.910496, z=0.275307),
    carla.Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000))
# 将车辆置于此位置
ego = world.spawn_actor(ego_bp, vehicle_spawn_points0)
# ego_bp.set_autopilot(False) 是否设置自动驾驶的开关

######################
# 通过代码控制车辆运动的部分
# 设置throttle和brake以及steer
throttle = 0
brake = 0
steer = 0
# 将控制量变为可输入的格式
act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
# 输入车辆
ego.apply_control(act)
# 运行一个时间dt
world.tick()

###########################
# 获取carla中运行后的车辆位置，速度
ego_trans = ego.get_transform()
ego_x = ego_trans.location.x
ego_y = ego_trans.location.y
ego_v = ego.get_velocity()

###########################
# 清空单位
actor_filters = ['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb',
                 'sensor.camera.semantic_segmentation', 'vehicle.*', 'controller.ai.walker', 'walker.*']


def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
        for actor in self.world.get_actors().filter(actor_filter):
            if actor.is_alive:
                if actor.type_id == 'controller.ai.walker':
                    actor.stop()
                actor.destroy()


############################
# 设置相机等的参数
camera_img = np.zeros((384, 216, 3), dtype=np.uint8)
camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
# Modify the attributes of the blueprint to set image resolution and field of view.
camera_bp.set_attribute('image_size_x', str(384))
camera_bp.set_attribute('image_size_y', str(216))
camera_bp.set_attribute('fov', '110')
# Set the time in seconds between sensor captures
camera_bp.set_attribute('sensor_tick', '0.02')
# 将camera置于车上
camera_sensor = world.spawn_actor(camera_bp, camera_trans, attach_to=ego)
# 获得camera的图像
location_flag = None
camera_sensor.listen(lambda data: get_camera_img(data))

# 获取图像的函数
    def get_camera_img(data):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        image = np.reshape(array, (data.height, data.width, 4))

        # Get the r channel
        sem = image[:, :, 2]
        # print(sem)
        m = len(sem[0, :])
        if location_flag == None:
            for i in range(len(sem[:, 0])):
                for j in range(int(m / 2)):
                    if sem[i][j + int(m / 2)] == 4:
                        location_flag = True
