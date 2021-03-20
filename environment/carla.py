from gym.spaces.box import Box
import numpy as np

import glob
import os
import sys
import time
import numpy as np
import cv2
import math
import queue
from PIL import Image

try:
    sys.path.append(glob.glob("/home/nugroho/Projects/Simulator/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg" % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class CarlaEnv():
    def __init__(self, im_height = 480, im_width = 480, im_preview = False, max_step = 512, index_pos = None):
        self.cur_step           = 0
        self.collision_hist     = []
        self.crossed_line_hist  = []
        self.actor_list         = []
        self.init_pos           = [
            [-149.1, -94.7, 89.3],
            [-152.3, -96.5, 90.8],
            [-141.9, -33.6, -89.0],
            [-145.5, -33.6, -89.6],
        ]

        self.im_height              = im_height
        self.im_width               = im_width
        self.im_preview             = im_preview
        self.max_step               = max_step
        self.index_pos              = index_pos

        self.observation_space      = Box(low = -1.0, high = 1.0, shape = (im_height, im_width))
        self.action_space           = Box(low = -1.0, high = 1.0, shape = (2, 1))

        client              = carla.Client('127.0.0.1', 2000)
        self.world          = client.get_world()
        blueprint_library   = self.world.get_blueprint_library()

        self.model_3        = blueprint_library.filter('model3')[0]
        self.rgb_cam        = blueprint_library.find('sensor.camera.rgb')
        self.col_detector   = blueprint_library.find('sensor.other.collision')
        self.crl_detector   = blueprint_library.find('sensor.other.lane_invasion')        

        self.rgb_cam.set_attribute('image_size_x', f'{im_height}')
        self.rgb_cam.set_attribute('image_size_y', f'{im_width}')

        settings                        = self.world.get_settings()
        settings.synchronous_mode       = True
        settings.fixed_delta_seconds    = 0.05
        self.world.apply_settings(settings)

        self.cam_queue  = queue.Queue()         
        
    def __del__(self):
        for actor in self.actor_list:
            actor.destroy()
        del self.actor_list[:]

    def __get_pos(self):
        if self.index_pos is not None:
            return self.init_pos[self.index_pos]
        else:
            idx_pos = np.random.randint(len(self.init_pos))
            return self.init_pos[idx_pos]

    def __process_image(self, image):
        # image.convert(carla.ColorConverter.CityScapesPalette)        
        
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, -1))
        i = i[:, :, :3]

        if self.im_preview:
            cv2.imshow('', i)
            cv2.waitKey(1)

        i = Image.fromarray(i)
        return i

    def __process_collision(self, event):
        self.collision_hist.append(event)

    def __process_crossed_line(self, event):
        self.crossed_line_hist.append(event)

    def __tick_env(self):
        self.world.tick()
        time.sleep(0.05)

    def is_discrete(self):
        return False

    def get_obs_dim(self):
        return self.im_height * self.im_width
            
    def get_action_dim(self):
        return 2

    def reset(self):
        for actor in self.actor_list:
            actor.destroy()
        del self.actor_list[:]        

        pos = self.__get_pos()
        pos = carla.Transform(carla.Location(x = pos[0], y = pos[1], z = 1.0), carla.Rotation(pitch = 0, yaw = pos[2], roll = 0))
        
        self.vehicle    = self.world.spawn_actor(self.model_3, pos)        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0, brake = 1.0, steer = 0))        
                
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, carla.Transform(carla.Location(x = 1.6, z = 1.7)), attach_to = self.vehicle)
        self.cam_sensor.listen(self.cam_queue.put)

        for _ in range(4):
            self.__tick_env()
        
        self.col_sensor = self.world.spawn_actor(self.col_detector, carla.Transform(), attach_to = self.vehicle)
        self.col_sensor.listen(lambda event: self.__process_collision(event))
        
        self.crl_sensor = self.world.spawn_actor(self.crl_detector, carla.Transform(), attach_to = self.vehicle)
        self.crl_sensor.listen(lambda event: self.__process_crossed_line(event))
        
        self.actor_list.append(self.cam_sensor)
        self.actor_list.append(self.col_sensor)
        self.actor_list.append(self.crl_sensor)
        self.actor_list.append(self.vehicle)

        self.cur_step = 0

        image = self.__process_image(self.cam_queue.get())
        del self.collision_hist[:]
        del self.crossed_line_hist[:] 
        
        return image, np.array([0])

    def step(self, action):
        prev_loc    = self.vehicle.get_location()

        steer       = -1 if action[0] < -1 else 1 if action[0] > 1 else action[0]
        throttle    = 0 if action[1] < 0 else 1 if action[1] > 1 else action[1]
        brake       = 0 if action[2] < 0 else 1 if action[2] > 1 else action[2]
        self.vehicle.apply_control(carla.VehicleControl(steer = float(steer), throttle = float(throttle), brake = float(brake)))

        self.__tick_env()
        self.cur_step   += 1

        v       = self.vehicle.get_velocity()
        kmh     = math.sqrt(v.x ** 2 + v.y ** 2)
                
        loc     = self.vehicle.get_location()
        dif_x   = loc.x - prev_loc.x if loc.x - prev_loc.x >= 0.05 else 0
        dif_y   = loc.y - prev_loc.y if loc.y - prev_loc.y >= 0.05 else 0
        dif_loc = math.sqrt(dif_x ** 2 + dif_y ** 2)

        done    = False
        reward  = (dif_loc * 10) - 0.1        
        
        image   = self.__process_image(self.cam_queue.get())
        if len(self.crossed_line_hist) > 0 or len(self.collision_hist) > 0 or loc.x >= -100 or loc.y >= -10 or self.cur_step >= self.max_step:
            done = True
        
        return image, np.array([kmh]), reward, done, None