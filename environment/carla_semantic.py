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
import ray

import carla
from environment.carla import CarlaEnv

class CarlaSemanticEnv(CarlaEnv):
    def __init__(self, im_height = 480, im_width = 480, im_preview = False, max_step = 512, index_pos = None):
        super().__init__(im_height, im_width, im_preview, max_step, index_pos)

        blueprint_library   = self.world.get_blueprint_library()
        self.rgb_cam        = blueprint_library.find('sensor.camera.semantic_segmentation')

        self.rgb_cam.set_attribute('image_size_x', f'{im_height}')
        self.rgb_cam.set_attribute('image_size_y', f'{im_width}')

    def _process_image(self, image):
        i = np.array(image.raw_data)        
        i = i.reshape((self.im_height, self.im_width, -1))
        i = i[:, :, 0]

        return i