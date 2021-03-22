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

from environment.carla import CarlaEnv

class CarlaSemanticEnv(CarlaEnv):
    def _process_image(self, image):
        i = np.array(image.raw_data)        
        i = i.reshape((self.im_height, self.im_width, -1))
        i = i[:, :, 0]

        return i